"""
Synthesiser: takes retrieved chunks and a query, streams an answer.

For large document sets (>20 chunks), uses map-reduce:
  MAP:    summarise each source file separately
  REDUCE: combine summaries into a final answer
"""

import os
from collections.abc import Generator

import ollama

from brain.search.language import language_instruction, detect_language

_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

_SYSTEM = """You are a personal knowledge assistant. Answer questions using \
ONLY the document excerpts provided.

Rules:
- PROCESS EVERY EXCERPT. Do not skip any. Read each one and extract relevant information.
- Give a COMPREHENSIVE answer that includes ALL relevant facts from ALL excerpts. \
If you find 20 articles, list all 20. If you find 10 events, list all 10. \
Do not summarise or truncate -- completeness is more important than brevity.
- The excerpts may be in Russian or English regardless of the question language. \
Read ALL excerpts carefully, including Russian ones.
- Russian names and facts are valid answers to English questions. \
Transliterate names: "Иван Петров" -> "Ivan Petrov".
- URLs followed by text mean the user read that page and noted their thoughts. \
For each URL found, list it as an article the user read with its note as a summary.
- CRITICAL -- DO NOT HALLUCINATE: If a topic is only briefly mentioned in the \
excerpts (e.g. "was reading a novel", "tried a hobby a few times"), report \
ONLY what is literally written. Do NOT add plot summaries, character details, \
invented memories, or any information from your training data. \
If the excerpts contain only a passing mention, say: \
"The documents only briefly mention X: [quote the exact text]. No further detail is recorded."
- Do not invent, estimate, or extrapolate facts not present in the excerpts. \
If asked for totals or estimates and the number is not stated explicitly, \
count what you can see and say "at least N" rather than guessing.
- Cite which document(s) the information came from.
"""


def _model_options() -> dict:
    opts = {}
    raw = os.getenv("OLLAMA_NUM_GPU_LAYERS")
    if raw:
        opts["num_gpu"] = int(raw)
    opts["num_predict"] = 2048
    return opts


def _stream_safe(stream) -> Generator[str, None, None]:
    """Stream tokens, stopping if the model drifts into CJK characters."""
    last = None
    cjk_count = 0
    total = 0
    for chunk in stream:
        content = chunk["message"]["content"]
        if content and content != last:
            cjk_count += sum(
                1
                for c in content
                if 0x4E00 <= ord(c) <= 0x9FFF or 0x3040 <= ord(c) <= 0x30FF
            )
            total += len(content)
            if total > 150 and cjk_count / total > 0.20:
                print(
                    f"[SYNTHESISER] CJK drift ({cjk_count}/{total} chars), stopping"
                )
                break
            yield content
        last = content


def synthesise(
    query: str,
    chunks: list[dict],
) -> Generator[str, None, None]:
    """Стримит ответ по токенам для запроса и найденных чанков."""
    lang = detect_language(query)
    estimated_tokens = len(chunks) * 400

    if estimated_tokens > 8000 and len(chunks) > 10:
        print(
            f"[SYNTHESISER] large context ({len(chunks)} chunks), using map-reduce"
        )
        yield from _map_reduce_synthesise(query, chunks, lang)
        return

    system = language_instruction(query) + "\n\n" + _SYSTEM
    user = _build_user_prompt(query, chunks, lang)

    stream = ollama.chat(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options=_model_options() or None,
        stream=True,
    )

    yield from _stream_safe(stream)


def _map_reduce_synthesise(
    query: str,
    chunks: list[dict],
    lang: str,
) -> Generator[str, None, None]:
    """
    Map-reduce synthesis for large document sets.

    MAP: For each source file, summarise what it says about the query.
    REDUCE: Combine all per-file summaries into a final answer.
    """
    # Group chunks by source file
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        fname = c["file_path"].replace("\\", "/").split("/")[-1]
        by_file.setdefault(fname, []).append(c)

    map_system = (
        language_instruction(query) + "\n\n"
        "Extract ONLY information literally present in the excerpts that is "
        "relevant to the question, and quote the relevant text. Do NOT add any "
        "facts, plot details, opinions, or knowledge not written in the "
        "excerpts. If a book/film/show/game is only briefly mentioned, say so "
        "and quote the exact mention. If nothing relevant is present, say "
        "'Nothing relevant found.'"
    )

    per_file_summaries = []
    for fname, file_chunks in by_file.items():
        # Identify high-relevance chunks (keyword-augmented)
        high_rel_indices = {
            c.get("chunk_index", -1)
            for c in file_chunks
            if c.get("score", 0) >= 0.90
        }

        # For each high-relevance chunk, include it plus neighbours (±3 chunks)
        # to capture the full section around it
        if high_rel_indices:
            included_indices = set()
            all_indices = {
                c.get("chunk_index", i): i for i, c in enumerate(file_chunks)
            }
            for idx in high_rel_indices:
                for offset in range(
                    -2, 6
                ):  # 2 before, 5 after (conclusions tend to run forward)
                    neighbour = idx + offset
                    if neighbour in all_indices:
                        included_indices.add(neighbour)

            priority_chunks = [
                c
                for c in file_chunks
                if c.get("chunk_index", -1) in included_indices
            ]
            # Also add top semantic chunks that aren't already included
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

        user_msg = f"{context}\nQuestion: {query}\n\nExtract all relevant information in {lang}:"

        try:
            resp = ollama.chat(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": map_system},
                    {"role": "user", "content": user_msg},
                ],
                options=_model_options() or None,
            )
            summary = resp["message"]["content"].strip()
            if summary and "nothing relevant" not in summary.lower():
                per_file_summaries.append(f"From {fname}:\n{summary}")
        except Exception as e:
            print(f"[MAP] failed for {fname}: {e}")
            continue

    if not per_file_summaries:
        yield "No relevant information was found in your documents for this query."
        return

    reduce_system = (
        language_instruction(query) + "\n\n"
        "You have summaries of relevant information from the user's documents. "
        "Combine them into a comprehensive answer. Include ALL facts mentioned. "
        "Do NOT add any information not present in the summaries -- no invented "
        "plot details, opinions, or outside knowledge. If the summaries show "
        "only brief mentions, say exactly that."
    )

    combined = "\n\n".join(per_file_summaries)
    reduce_user = (
        f"Summaries from documents:\n\n{combined}\n\n"
        f"Question: {query}\n\nAnswer in {lang}:"
    )

    stream = ollama.chat(
        model=_MODEL,
        messages=[
            {"role": "system", "content": reduce_system},
            {"role": "user", "content": reduce_user},
        ],
        options=_model_options() or None,
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
    parts.append(
        f"Answer in {lang} ONLY. Do not use any other language.\n"
        f"The excerpts above are in Russian. Read them carefully and extract "
        f"the answer to the question. Personal reflections, observations, and "
        f"opinions written in Russian diary style ARE valid answers to English "
        f"questions about conclusions, thoughts, or takeaways.\n"
        f"Be specific, direct, and comprehensive -- include ALL relevant content."
    )

    return "\n".join(parts)
