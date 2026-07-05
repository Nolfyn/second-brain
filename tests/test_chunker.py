from brain.index.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_one_chunk():
    text = "This is a short sentence. Here is another one."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_long_text_splits_into_multiple_chunks():
    # Build text well over the chunk size threshold
    sentence = (
        "This is a moderately long sentence about stoicism and philosophy. "
    )
    text = sentence * 50  # ~3500 chars, should split into 2+ chunks
    chunks = chunk_text(text)
    assert len(chunks) > 1


def test_chunks_have_sequential_indices():
    sentence = (
        "This is a moderately long sentence about stoicism and philosophy. "
    )
    text = sentence * 50
    chunks = chunk_text(text)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_no_sentence_is_cut_in_half():
    # Every chunk should end with sentence-ending punctuation
    sentence = "This is a complete sentence about deep work. "
    text = sentence * 50
    chunks = chunk_text(text)
    for chunk in chunks:
        stripped = chunk.text.strip()
        assert stripped[-1] in ".!?"


def test_russian_text_chunks_correctly():
    text = (
        "Это тестовое предложение на русском языке. "
        "Семантический поиск должен работать одинаково хорошо "
        "как с русским, так и с английским текстом. "
        "Это третье предложение для проверки токенизации."
    )
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    assert "русском" in chunks[0].text
