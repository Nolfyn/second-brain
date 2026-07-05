"""
Tests for the LangChain parser integration.
These use in-memory temp files so no real documents are needed.
"""

import tempfile
from pathlib import Path

import pytest

from brain.parsers.langchain_parser import LangChainParser


def write_temp(suffix: str, content: str, encoding: str = "utf-8") -> Path:
    f = tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, mode="w", encoding=encoding
    )
    f.write(content)
    f.close()
    return Path(f.name)


def test_txt_parser_returns_text():
    path = write_temp(
        ".txt", "First sentence. Second sentence about philosophy."
    )
    parser = LangChainParser()
    result = parser.parse(path)
    assert result.file_type == "txt"
    assert "philosophy" in result.raw_text
    path.unlink()


def test_txt_parser_handles_russian():
    path = write_temp(".txt", "Это тестовое предложение на русском языке.")
    parser = LangChainParser()
    result = parser.parse(path)
    assert "русском" in result.raw_text
    path.unlink()


def test_csv_parser_returns_row_content():
    path = write_temp(".csv", "title,year\nStalker,1979\nPersona,1966\n")
    parser = LangChainParser()
    result = parser.parse(path)
    assert "Stalker" in result.raw_text
    assert "Persona" in result.raw_text
    path.unlink()


def test_html_parser_extracts_body_text():
    html = (
        "<html><body><h1>My Notes</h1><p>Some content here.</p></body></html>"
    )
    path = write_temp(".html", html)
    parser = LangChainParser()
    result = parser.parse(path)
    assert "content" in result.raw_text.lower()
    path.unlink()


def test_unsupported_extension_raises():
    parser = LangChainParser()
    with pytest.raises(ValueError, match="does not support"):
        parser._get_loader(Path("file.xyz"), ".xyz")


def test_dispatcher_routes_txt_to_langchain():
    from brain.parsers import get_parser
    from brain.parsers.langchain_parser import LangChainParser

    path = Path("anything.txt")
    # get_parser returns an instance; check its type without actually parsing
    parser = get_parser(path)
    assert isinstance(parser, LangChainParser)


def test_dispatcher_routes_csv_to_langchain():
    from brain.parsers import get_parser
    from brain.parsers.langchain_parser import LangChainParser

    assert isinstance(get_parser(Path("data.csv")), LangChainParser)


def test_dispatcher_routes_epub_to_langchain():
    from brain.parsers import get_parser
    from brain.parsers.langchain_parser import LangChainParser

    assert isinstance(get_parser(Path("book.epub")), LangChainParser)
