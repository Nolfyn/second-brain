from dataclasses import dataclass

import nltk

# Download once, cached after first run
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


@dataclass
class Chunk:
    text: str
    chunk_index: int  # position of this chunk within its source document


# Rough token budget per chunk. We approximate tokens as ~4 chars/token,
# which is a reasonable average across English and Russian.
_TARGET_CHUNK_CHARS = 1600  # ~400 tokens
_OVERLAP_CHARS = 200  # ~50 tokens


def chunk_text(text: str) -> list[Chunk]:
    """
    Split text into overlapping chunks along sentence boundaries.
    Never cuts a sentence in half. Each chunk overlaps the previous
    one by roughly _OVERLAP_CHARS so ideas spanning a boundary aren't lost.
    """
    text = text.strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if (
            current_len + sentence_len > _TARGET_CHUNK_CHARS
            and current_sentences
        ):
            # Flush the current chunk
            chunk_str = " ".join(current_sentences)
            chunks.append(Chunk(text=chunk_str, chunk_index=len(chunks)))

            # Start the next chunk with overlap: carry over trailing
            # sentences from the end of the previous chunk
            current_sentences = _take_overlap(current_sentences)
            current_len = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_len += sentence_len

    # Flush the final chunk
    if current_sentences:
        chunk_str = " ".join(current_sentences)
        chunks.append(Chunk(text=chunk_str, chunk_index=len(chunks)))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """
    Split into sentences. nltk's punkt tokenizer handles both
    English and Russian punctuation conventions reasonably well.
    Falls back to paragraph splitting if nltk fails for any reason.
    """
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        return [p.strip() for p in text.split("\n") if p.strip()]


def _take_overlap(sentences: list[str]) -> list[str]:
    """Return the trailing sentences that fit within the overlap budget."""
    overlap: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > _OVERLAP_CHARS:
            break
        overlap.insert(0, sentence)
        total += len(sentence)
    return overlap
