"""Тесты извлечения структурных секций из docx и txt."""

from pathlib import Path

from brain.parsers.docx_parser import DocxParser
from brain.parsers.text_parser import TextParser


def test_docx_headings_become_sections(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("Вступление без заголовка.")
    doc.add_heading("Поездка", level=1)
    doc.add_paragraph("Текст про поездку. Ещё абзац.")
    doc.add_heading("Выводы", level=2)
    doc.add_paragraph("Итоговые мысли.")
    path = tmp_path / "note.docx"
    doc.save(path)

    result = DocxParser().parse(path)
    titles = [s.title for s in result.sections]
    assert titles[0] is None  # текст до первого заголовка
    assert "Поездка" in titles
    assert "Выводы" in titles
    by_title = {s.title: s.text for s in result.sections}
    assert "Текст про поездку" in by_title["Поездка"]
    assert "Итоговые мысли" in by_title["Выводы"]


def test_docx_without_headings_has_no_sections(tmp_path):
    from docx import Document

    doc = Document()
    doc.add_paragraph("Просто текст без структуры.")
    path = tmp_path / "plain.docx"
    doc.save(path)

    result = DocxParser().parse(path)
    assert result.sections == []


def test_txt_date_lines_become_sections(tmp_path):
    path = tmp_path / "journal.txt"
    path.write_text(
        "Заметка до даты.\n"
        "15.03.2017\n"
        "События этого дня. Подробности.\n"
        "02.07.2017\n"
        "Другой день.",
        encoding="utf-8",
    )

    result = TextParser().parse(path)
    titles = [s.title for s in result.sections]
    assert "15.03.2017" in titles
    assert "02.07.2017" in titles
    by_title = {s.title: s.text for s in result.sections}
    assert "События этого дня" in by_title["15.03.2017"]


def test_txt_without_dates_has_no_sections(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("Просто текст. Без дат.", encoding="utf-8")

    result = TextParser().parse(path)
    assert result.sections == []
