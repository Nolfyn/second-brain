"""Тесты парсера простых текстовых форматов."""

import tempfile
from pathlib import Path

import pytest

from brain.parsers.text_parser import TextParser


def write_temp(suffix: str, content: str, encoding: str = "utf-8") -> Path:
    f = tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, mode="w", encoding=encoding
    )
    f.write(content)
    f.close()
    return Path(f.name)


def test_txt_parser_returns_text():
    path = write_temp(".txt", "First sentence. Second about philosophy.")
    result = TextParser().parse(path)
    assert result.file_type == "txt"
    assert "philosophy" in result.raw_text
    path.unlink()


def test_txt_parser_handles_russian():
    path = write_temp(".txt", "Это тестовое предложение на русском языке.")
    result = TextParser().parse(path)
    assert "русском" in result.raw_text
    path.unlink()


def test_txt_parser_handles_windows1251():
    path = write_temp(".txt", "Старый файл в cp1251.", encoding="cp1251")
    result = TextParser().parse(path)
    assert "cp1251" in result.raw_text
    path.unlink()


def test_csv_parser_returns_row_content():
    path = write_temp(".csv", "title,year\nStalker,1979\nPersona,1966\n")
    result = TextParser().parse(path)
    assert "Stalker" in result.raw_text
    assert "Persona" in result.raw_text
    path.unlink()


def test_html_parser_extracts_body_text():
    html = "<html><body><h1>My Notes</h1><p>Some content here.</p></body></html>"
    path = write_temp(".html", html)
    result = TextParser().parse(path)
    assert "content" in result.raw_text.lower()
    path.unlink()


def test_epub_parser_extracts_chapter_text(tmp_path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title("Test Book")
    book.set_language("en")
    chapter = epub.EpubHtml(title="Ch1", file_name="ch1.xhtml")
    chapter.content = "<h1>Chapter 1</h1><p>Unique epub sentence.</p>"
    book.add_item(chapter)
    book.spine = [chapter]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    path = tmp_path / "book.epub"
    epub.write_epub(str(path), book)

    result = TextParser().parse(path)
    assert "Unique epub sentence" in result.raw_text


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="does not support"):
        TextParser().parse(Path("file.xyz"))


@pytest.mark.parametrize(
    "name",
    ["notes.txt", "data.csv", "page.html", "book.epub", "slides.pptx"],
)
def test_dispatcher_routes_simple_formats_to_text_parser(name):
    from brain.parsers import get_parser

    assert isinstance(get_parser(Path(name)), TextParser)
