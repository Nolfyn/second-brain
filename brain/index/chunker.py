"""Чанкинг: разбиение предложений с учётом структурных секций.

Секция - граница заметки (заголовок, дата). Чанкинг старается не резать
секцию: мелкие секции склеиваются до целевого размера, переполненная
секция делится по предложениям, но все её части несут её заголовок -
так синтез потом группирует документ по смысловым частям.
"""

from dataclasses import dataclass

import nltk

from brain.sections import Section

# Загружается один раз, дальше из кэша
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


@dataclass
class Chunk:
    text: str
    chunk_index: int  # позиция в исходном документе
    section: str | None = None  # заголовок секции (первой, если склейка)


# Оценка ~4 символа на токен - средняя для русского и английского
_TARGET_CHUNK_CHARS = 1600  # ~400 токенов
_OVERLAP_CHARS = 200  # ~50 токенов


def chunk_text(
    text: str, sections: list[Section] | None = None
) -> list[Chunk]:
    """Разбивает текст на чанки. Секции не режутся, если помещаются целиком."""
    if sections is not None:
        return _chunk_by_sections(sections)

    text = text.strip()
    if not text:
        return []
    sentences = _split_sentences(text)
    if not sentences:
        return []
    return _pack_sentences(sentences)


def _chunk_by_sections(sections: list[Section]) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending: list[str] = []
    pending_len = 0
    pending_title: str | None = None

    def flush() -> None:
        nonlocal pending, pending_len, pending_title
        if pending:
            chunks.append(
                Chunk(
                    text=" ".join(pending),
                    chunk_index=len(chunks),
                    section=pending_title or "",
                )
            )
            pending, pending_len, pending_title = [], 0, None

    for section in sections:
        body = section.text.strip()
        if not body:
            continue
        if len(body) > _TARGET_CHUNK_CHARS:
            flush()
            for part in _split_oversized(body):
                chunks.append(
                    Chunk(
                        text=part,
                        chunk_index=len(chunks),
                        section=section.title or "",
                    )
                )
            continue
        if pending and pending_len + len(body) > _TARGET_CHUNK_CHARS:
            flush()
        if not pending:
            pending_title = section.title
        pending.append(body)
        pending_len += len(body) + 1

    flush()
    return chunks


def _split_oversized(text: str) -> list[str]:
    """Переполненная секция делится по предложениям как обычный текст."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text]
    return [c.text for c in _pack_sentences(sentences)]


def _pack_sentences(sentences: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if (
            current_len + sentence_len > _TARGET_CHUNK_CHARS
            and current
        ):
            chunks.append(
                Chunk(
                    text=" ".join(current),
                    chunk_index=len(chunks),
                )
            )
            current = _take_overlap(current)
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(
            Chunk(text=" ".join(current), chunk_index=len(chunks))
        )
    return chunks


def _split_sentences(text: str) -> list[str]:
    """nltk punkt нормально работает и с русской, и с английской пунктуацией."""
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return [p.strip() for p in text.split("\n") if p.strip()]


def _take_overlap(sentences: list[str]) -> list[str]:
    """Хвостовые предложения предыдущего чанка в бюджете перекрытия."""
    overlap: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > _OVERLAP_CHARS:
            break
        overlap.insert(0, sentence)
        total += len(sentence)
    return overlap
