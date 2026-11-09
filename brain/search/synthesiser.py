"""Синтез ответа из найденных чанков.

Для больших наборов (>20 чанков) - map-reduce: map идёт по секциям
документов (заголовки, даты), reduce объединяет сводки в финальный ответ.
"""

import logging
from collections.abc import Generator

from brain.llm import get_llm
from brain.search.language import language_instruction, detect_language
from brain.search import prompts

log = logging.getLogger(__name__)

# Пороги map-reduce: ~400 токенов на чанк, лимит контекста ~8000 токенов
_TOKENS_PER_CHUNK = 400
_MAP_REDUCE_TOKEN_BUDGET = 8000
_MAP_REDUCE_MIN_CHUNKS = 10
# Символьный бюджет одного map-вызова (~6k токенов); переполненная
# секция делится на батчи - без потери середины
_SECTION_BATCH_CHARS = 24000

_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3040, 0x30FF))


def _stream_safe(stream) -> Generator[str, None, None]:
    """Обрывает поток, если модель начала генерировать CJK-мусор."""
    cjk_count = 0
    total = 0
    for content in stream:
        cjk_count += sum(
            1
            for c in content
            if any(lo <= ord(c) <= hi for lo, hi in _CJK_RANGES)
        )
        total += len(content)
        if total > 150 and cjk_count / total > 0.20:
            log.warning("CJK drift, stopping")
            return
        yield content


def synthesise(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Стримит ответ по токенам для запроса и найденных чанков.

    history - прошлые сообщения чата (user/assistant), уходят в контекст.
    """
    lang = detect_language(query)
    estimated_tokens = len(chunks) * _TOKENS_PER_CHUNK

    if (
        estimated_tokens > _MAP_REDUCE_TOKEN_BUDGET
        and len(chunks) > _MAP_REDUCE_MIN_CHUNKS
    ):
        log.info("large context (%d chunks), using map-reduce", len(chunks))
        yield from _map_reduce_synthesise(query, chunks, lang, history)
        return

    llm = get_llm()
    system = (
        language_instruction(query)
        + "\n\n"
        + prompts.get_prompts()["synthesis_system"]
    )
    user = _build_user_prompt(query, chunks, lang)

    messages = [{"role": "system", "content": system}]
    messages += history or []
    messages.append({"role": "user", "content": user})

    # Низкая температура: фактический синтез, а не творчество -
    # иначе 7B-модели склонны к отказам и отклонениям между прогонами
    stream = llm.chat(messages, stream=True, temperature=0.3)
    yield from _stream_safe(stream)


def _map_reduce_synthesise(
    query: str,
    chunks: list[dict],
    lang: str,
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """MAP: сводка по каждой секции каждого файла. REDUCE: общий ответ."""
    llm = get_llm()
    map_system = (
        language_instruction(query) + "\n\n" + prompts.get_prompts()["map_system"]
    )

    summaries = []
    for fname, file_chunks in _group_by_file(chunks).items():
        for section, sec_chunks in _group_by_section(fname, file_chunks).items():
            batches = _section_batches(sec_chunks)
            for bi, batch in enumerate(batches):
                label = (
                    section
                    if len(batches) == 1
                    else f"{section} (часть {bi + 1})"
                )
                context = "".join(
                    f"[Excerpt {i}]\n{c['text']}\n\n"
                    for i, c in enumerate(batch, 1)
                )
                user_msg = (
                    f"File: {fname}\nSection: {label}\n\n"
                    + context
                    + f"\nQuestion: {query}\n\n"
                    f"Extract all relevant information in {lang}:"
                )
                try:
                    summary = llm.chat(
                        [
                            {"role": "system", "content": map_system},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.3,
                    ).strip()
                    if summary and "nothing relevant" not in summary.lower():
                        summaries.append(
                            f"From {fname} ({label}):\n{summary}"
                        )
                except Exception as e:
                    log.warning(
                        "map failed for %s / %s: %s", fname, label, e
                    )

    if not summaries:
        yield "No relevant information was found in your documents for this query."
        return

    reduce_system = (
        language_instruction(query)
        + "\n\n"
        + prompts.get_prompts()["reduce_system"]
    )
    combined = "\n\n".join(summaries)
    reduce_user = (
        f"Summaries from documents:\n\n{combined}\n\n"
        f"Question: {query}\n\nAnswer in {lang}:"
    )

    stream = llm.chat(
        [{"role": "system", "content": reduce_system}]
        + (history or [])
        + [{"role": "user", "content": reduce_user}],
        stream=True,
        temperature=0.3,
    )
    yield from _stream_safe(stream)


def _section_batches(sec_chunks: list[dict]) -> list[list[dict]]:
    """Делит чанки секции на батчи по символьному бюджету.

    Длинная секция без структуры (дневник на 60+ чанков) проходит
    через map целиком, батч за батчом - середина не вырезается.
    """
    batches = []
    current = []
    current_len = 0
    for c in sec_chunks:
        if current and current_len + len(c["text"]) > _SECTION_BATCH_CHARS:
            batches.append(current)
            current = []
            current_len = 0
        current.append(c)
        current_len += len(c["text"])
    if current:
        batches.append(current)
    return batches


def _group_by_file(chunks: list[dict]) -> dict[str, list[dict]]:
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        fname = c["file_path"].replace("\\", "/").split("/")[-1]
        by_file.setdefault(fname, []).append(c)
    return by_file


def _group_by_section(
    fname: str, file_chunks: list[dict]
) -> dict[str, list[dict]]:
    """Секция без заголовка ("") группируется под именем файла."""
    by_section: dict[str, list[dict]] = {}
    for c in file_chunks:
        key = c.get("section") or fname
        by_section.setdefault(key, []).append(c)
    return by_section


def _build_user_prompt(
    query: str,
    chunks: list[dict],
    lang: str = "English",
) -> str:
    parts = []

    if chunks:
        parts.append(
            f"DOCUMENT EXCERPTS ({len(chunks)} total -- process ALL of them):"
        )
        for i, c in enumerate(chunks, 1):
            fname = c["file_path"].replace("\\", "/").split("/")[-1]
            parts.append(f"\n[Excerpt {i} from {fname}]\n{c['text']}")

    if not chunks:
        parts.append("No relevant content was found in your documents.")

    parts.append(f"\nQUESTION: {query}")
    parts.append(prompts.render("synthesis_user_tail", lang=lang))

    return "\n".join(parts)
