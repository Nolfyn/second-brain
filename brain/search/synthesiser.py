"""Синтез ответа из найденных чанков.

Для больших наборов (>20 чанков) — map-reduce: сначала сводка по каждому
файлу, затем объединение сводок в финальный ответ.
"""

from collections.abc import Generator

from brain.llm import get_llm
from brain.search.language import language_instruction, detect_language
from brain.search import prompts

# Пороги map-reduce: ~400 токенов на чанк, лимит контекста ~8000 токенов
_TOKENS_PER_CHUNK = 400
_MAP_REDUCE_TOKEN_BUDGET = 8000
_MAP_REDUCE_MIN_CHUNKS = 10

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
            print("[SYNTHESISER] CJK drift, stopping")
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
        print(
            f"[SYNTHESISER] large context ({len(chunks)} chunks), using map-reduce"
        )
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
    """MAP: сводка по каждому файлу. REDUCE: объединение сводок в ответ."""
    llm = get_llm()

    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        fname = c["file_path"].replace("\\", "/").split("/")[-1]
        by_file.setdefault(fname, []).append(c)

    map_system = (
        language_instruction(query) + "\n\n" + prompts.get_prompts()["map_system"]
    )

    per_file_summaries = []
    for fname, file_chunks in by_file.items():
        # Высокорелевантные чанки (keyword) с соседями (±3), плюс топ семантики
        high_rel_indices = {
            c.get("chunk_index", -1)
            for c in file_chunks
            if c.get("score", 0) >= 0.90
        }

        if high_rel_indices:
            included_indices = set()
            all_indices = {
                c.get("chunk_index", i): i for i, c in enumerate(file_chunks)
            }
            for idx in high_rel_indices:
                for offset in range(-2, 6):  # 2 до, 5 после
                    neighbour = idx + offset
                    if neighbour in all_indices:
                        included_indices.add(neighbour)

            priority_chunks = [
                c
                for c in file_chunks
                if c.get("chunk_index", -1) in included_indices
            ]
            other_chunks = [
                c
                for c in file_chunks
                if c.get("chunk_index", -1) not in included_indices
                and c.get("score", 0) >= 0.45
            ]
            map_chunks = (
                priority_chunks
                + other_chunks[: max(0, 10 - len(priority_chunks))]
            )
        else:
            map_chunks = file_chunks[:10]

        context = f"File: {fname}\n\n"
        for i, c in enumerate(map_chunks, 1):
            context += f"[Excerpt {i}]\n{c['text']}\n\n"

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
                per_file_summaries.append(f"From {fname}:\n{summary}")
        except Exception as e:
            print(f"[MAP] failed for {fname}: {e}")
            continue

    if not per_file_summaries:
        yield "No relevant information was found in your documents for this query."
        return

    reduce_system = (
        language_instruction(query)
        + "\n\n"
        + prompts.get_prompts()["reduce_system"]
    )
    combined = "\n\n".join(per_file_summaries)
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
