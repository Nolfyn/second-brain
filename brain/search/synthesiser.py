"""Синтез ответа из найденных чанков.

Для больших наборов (>20 чанков) — map-reduce: map идёт по секциям
документов (заголовки, даты), reduce объединяет сводки в финальный ответ.
Так длинная однотемная заметка обрабатывается целиком — модель не видит
только начало и конец.
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
# Бюджет контекста одного map-вызова (символы)
_MAP_SECTION_CHAR_BUDGET = 6000

_CJK_RANGES = ((0x4E00, 0x9FFF), (0x3040, 0x30FF))


def _stream_safe(stream) -> Generator[str, None, None]:
    """Обрывает стрим, если модель начала генерировать CJK-мусор."""
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
) -> Generator[str, None, None]:
    """Стримит ответ по токенам для запроса и найденных чанков."""
    lang = detect_language(query)
    estimated_tokens = len(chunks) * _TOKENS_PER_CHUNK

    if (
        estimated_tokens > _MAP_REDUCE_TOKEN_BUDGET
        and len(chunks) > _MAP_REDUCE_MIN_CHUNKS
    ):
        log.info("large context (%d chunks), using map-reduce", len(chunks))
        yield from _map_reduce_synthesise(query, chunks, lang)
        return

    llm = get_llm()
    system = (
        language_instruction(query)
        + "\n\n"
        + prompts.get_prompts()["synthesis_system"]
    )
    user = _build_user_prompt(query, chunks, lang)

    stream = llm.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
    )
    yield from _stream_safe(stream)


def _map_reduce_synthesise(
    query: str,
    chunks: list[dict],
    lang: str,
) -> Generator[str, None, None]:
    """MAP: сводка по каждой секции каждого файла. REDUCE: общий ответ."""
    llm = get_llm()
    map_system = (
        language_instruction(query) + "\n\n" + prompts.get_prompts()["map_system"]
    )

    summaries = []
    for fname, file_chunks in _group_by_file(chunks).items():
        for section, sec_chunks in _group_by_section(fname, file_chunks).items():
            context = f"File: {fname}\nSection: {section}\n\n"
            budget = _MAP_SECTION_CHAR_BUDGET
            for i, c in enumerate(sec_chunks, 1):
                if budget <= 0:
                    break
                context += f"[Excerpt {i}]\n{c['text'][:budget]}\n\n"
                budget -= len(c["text"])

            user_msg = (
                f"{context}\nQuestion: {query}\n\n"
                f"Extract all relevant information in {lang}:"
            )
            try:
                summary = llm.chat(
                    [
                        {"role": "system", "content": map_system},
                        {"role": "user", "content": user_msg},
                    ]
                ).strip()
                if summary and "nothing relevant" not in summary.lower():
                    summaries.append(f"From {fname} ({section}):\n{summary}")
            except Exception as e:
                log.warning("map failed for %s / %s: %s", fname, section, e)

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
        [
            {"role": "system", "content": reduce_system},
            {"role": "user", "content": reduce_user},
        ],
        stream=True,
    )
    yield from _stream_safe(stream)


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
