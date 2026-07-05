"""
Parser tests.

Required sample files in tests/fixtures/:
  sample_plain.docx       — an unprotected Word doc with some text
  sample_protected.docx   — a Word doc protected with the password in your .env
  sample_text.pdf         — a normal text-based PDF
  sample_scanned.pdf      — a scanned/image-based PDF (can use any scanned doc)
  sample_legacy.doc       — an old .doc file (optional, needs LibreOffice)

Create them manually or use the helper script: python tests/make_fixtures.py
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from brain.parsers import get_parser, SUPPORTED_EXTENSIONS
from brain.parsers.base import ParsedDocument

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse(filename: str) -> ParsedDocument:
    path = FIXTURES / filename
    parser = get_parser(path)
    return parser.parse(path)


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------


def test_supported_extensions_contains_expected():
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".doc" in SUPPORTED_EXTENSIONS
    assert ".pdf" in SUPPORTED_EXTENSIONS


def test_dispatcher_raises_on_unknown_extension():
    from brain.parsers.dispatcher import get_parser as gp

    with pytest.raises(ValueError, match="No parser available"):
        gp(Path("somefile.xyz"))


# ---------------------------------------------------------------------------
# DOCX parser tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (FIXTURES / "sample_plain.docx").exists(),
    reason="fixtures/sample_plain.docx not found — run tests/make_fixtures.py",
)
def test_docx_plain_returns_text():
    result = parse("sample_plain.docx")
    assert result.file_type == "docx"
    assert len(result.raw_text) > 10
    assert result.raw_text.strip() != ""


@pytest.mark.skipif(
    not (FIXTURES / "sample_protected.docx").exists(),
    reason="fixtures/sample_protected.docx not found",
)
def test_docx_protected_decrypts_correctly():
    result = parse("sample_protected.docx")
    assert len(result.raw_text) > 10


@pytest.mark.skipif(
    not (FIXTURES / "sample_plain.docx").exists(),
    reason="fixtures/sample_plain.docx not found",
)
def test_docx_result_is_valid_unicode():
    result = parse("sample_plain.docx")
    # Should not raise — all characters must be valid UTF-8
    result.raw_text.encode("utf-8")


# ---------------------------------------------------------------------------
# PDF parser tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (FIXTURES / "sample_text.pdf").exists(),
    reason="fixtures/sample_text.pdf not found",
)
def test_pdf_text_based_extracts_text():
    result = parse("sample_text.pdf")
    assert result.file_type == "pdf"
    assert len(result.raw_text) > 10


@pytest.mark.skipif(
    not (FIXTURES / "sample_scanned.pdf").exists(),
    reason="fixtures/sample_scanned.pdf not found — scanned PDF required",
)
def test_pdf_scanned_falls_back_to_ocr():
    # This test will be slow (OCR is slow) — expected
    result = parse("sample_scanned.pdf")
    assert len(result.raw_text) > 10


# ---------------------------------------------------------------------------
# DOC parser tests (requires LibreOffice)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (FIXTURES / "sample_legacy.doc").exists(),
    reason="fixtures/sample_legacy.doc not found",
)
def test_doc_legacy_converts_and_extracts():
    result = parse("sample_legacy.doc")
    assert result.file_type == "doc"
    assert len(result.raw_text) > 10


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_parse_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        parse("does_not_exist.docx")
