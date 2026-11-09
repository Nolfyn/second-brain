"""Тесты чанкера: базовое разбиение по предложениям и структурные секции."""

from brain.index.chunker import Section, chunk_text


def _sentences(n, base="Предложение номер"):
    return " ".join(f"{base} {i}." for i in range(n))


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_one_chunk():
    chunks = chunk_text("This is a short sentence. Here is another one.")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].section is None


def test_long_text_splits_into_multiple_chunks():
    sentence = "This is a moderately long sentence about stoicism and philosophy. "
    chunks = chunk_text(sentence * 50)
    assert len(chunks) > 1


def test_chunks_have_sequential_indices():
    sentence = "This is a moderately long sentence about stoicism and philosophy. "
    chunks = chunk_text(sentence * 50)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_no_sentence_is_cut_in_half():
    sentence = "This is a complete sentence about deep work. "
    chunks = chunk_text(sentence * 50)
    for chunk in chunks:
        assert chunk.text.strip()[-1] in ".!?"


# --- структурные секции ---


def test_small_sections_merge_into_one_chunk():
    sections = [
        Section(title="15.03.2017", text=_sentences(2)),
        Section(title="16.03.2017", text=_sentences(2)),
        Section(title="17.03.2017", text=_sentences(2)),
    ]
    chunks = chunk_text("", sections=sections)
    assert len(chunks) == 1
    assert chunks[0].section == "15.03.2017"
    for s in sections:
        assert s.text in chunks[0].text


def test_oversized_section_is_split_with_same_title():
    big_text = _sentences(80)  # значительно больше целевого размера чанка
    sections = [
        Section(title="Начало", text=_sentences(3)),
        Section(title="Огромная секция", text=big_text),
        Section(title="Конец", text=_sentences(3)),
    ]
    chunks = chunk_text("", sections=sections)
    assert len(chunks) > 3
    huge = [c for c in chunks if c.section == "Огромная секция"]
    assert len(huge) > 1
    # части огромной секции идут подряд и не перемешиваются с соседями
    texts = [c.section for c in chunks]
    assert texts[0] == "Начало"
    assert texts[-1] == "Конец"


def test_sections_without_titles_use_fallback():
    sections = [Section(title=None, text=_sentences(2))]
    chunks = chunk_text("", sections=sections)
    assert chunks[0].section == ""


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
