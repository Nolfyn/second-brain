"""Тесты извлечения сигналов из запроса."""

from brain.search.signals import (
    extract_entities,
    extract_keywords,
    extract_year,
    find_named_file,
    find_topic_scope,
)

INDEXED = [
    "C:/docs/Notes.docx",
    "C:/docs/Games.docx",
    "C:/docs/china/China.docx",
    "C:/docs/Egypt.docx",
    "C:/docs/J 2017.docx",
    "C:/docs/Sweden.docx",
]


def test_extracts_year():
    assert extract_year("Чем я занимался в 2017 году?") == "2017"
    assert extract_year("what happened in 1999") == "1999"


def test_no_year():
    assert extract_year("no year here") is None
    assert extract_year("число 12345") is None


def test_extracts_url():
    kws = extract_keywords("read https://gwern.net/zeo and notes")
    assert "https://gwern.net/zeo" in kws


def test_extracts_domain():
    assert "vk.com" in extract_keywords("статьи с vk.com я читал")
    assert "example.org" in extract_keywords("found on example.org today")


def test_extracts_quoted_phrase():
    kws = extract_keywords("What did I mention about 'Honkai: Star Rail'?")
    assert "Honkai: Star Rail" in kws


def test_extracts_bare_site():
    assert "gwern" in extract_keywords("What articles have I read from gwern?")


def test_bare_site_skips_function_words():
    assert extract_keywords("from here to there") == []


def test_quoted_filename_not_a_keyword():
    kws = extract_keywords('Check "Games.docx" please')
    assert "Games.docx" not in kws


def test_find_named_file_by_extension():
    assert (
        find_named_file("Check the file Games.docx", INDEXED) == "Games.docx"
    )


def test_find_named_file_by_quoted():
    assert find_named_file('что в "Notes.docx"', INDEXED) == "Notes.docx"


def test_find_named_file_none():
    assert find_named_file("nothing specific", INDEXED) is None


def test_topic_scope_with_reflection_trigger():
    topic = find_topic_scope(
        "What were my main conclusions after my trip to China?", INDEXED
    )
    assert topic == "China"


def test_topic_scope_none_without_trigger():
    assert find_topic_scope("trip to China notes", INDEXED) is None


def test_topic_scope_no_cross_language_folder_match():
    # без словаря переводов русское слово не сматчится с английской папкой —
    # такой запрос уходит в семантический путь
    assert (
        find_topic_scope("Сделай выводы по моим заметкам о Швеции.", INDEXED)
        is None
    )


def test_entities_keep_capitalized_content_words():
    entities = extract_entities(
        "What did I write about museums in Stockholm?",
        is_common=lambda w: False,
    )
    assert "Stockholm" in entities
    assert "What" not in entities
    assert "did" not in entities


def test_entities_keep_lowercase_rare_words():
    entities = extract_entities(
        "Where did I write about Qyzylorda?",
        is_common=lambda w: False,
    )
    assert "Qyzylorda" in entities


def test_entities_drop_corpus_common_words():
    entities = extract_entities(
        "notes about music and festivals",
        is_common=lambda w: w in {"music", "notes"},
    )
    assert "music" not in entities
    assert "notes" not in entities
    assert "festivals" in entities
