import io
import os
import re
from pathlib import Path

import msoffcrypto
from docx import Document

from brain.sections import Section
from .base import BaseParser, ParsedDocument

_HEADING_STYLE = re.compile(r"^(Heading|Заголовок)\s*\d*$", re.IGNORECASE)


class DocxParser(BaseParser):
    """
    Парсит .docx в plain text.
    Парольные файлы расшифровывает через DOC_PASSWORD — в памяти,
    без записи расшифрованного текста на диск.
    """

    def __init__(self):
        self._password = os.getenv("DOC_PASSWORD", "")

    def parse(self, path: Path) -> ParsedDocument:
        self._check_exists(path)

        try:
            doc = self._open_document(path)
            return ParsedDocument(
                file_path=str(path),
                file_type="docx",
                raw_text=self._extract_text(doc),
                sections=self._extract_sections(doc),
            )
        except Exception as e:
            raise ValueError(f"Failed to parse {path.name}: {e}") from e

    def _open_document(self, path: Path) -> Document:
        """Сначала открываем как обычный docx, при неудаче — расшифровка."""
        try:
            return Document(str(path))
        except Exception:
            if not self._password:
                raise ValueError(
                    f"{path.name} could not be opened. "
                    "If it's password-protected, "
                    "set DOC_PASSWORD in your .env file. "
                )
            return self._decrypt(path)

    def _decrypt(self, path: Path) -> Document:
        """Расшифровка Office-файла в буфер в памяти."""
        with open(path, "rb") as f:
            office_file = msoffcrypto.OfficeFile(f)
            office_file.load_key(password=self._password)
            decrypted_buffer = io.BytesIO()
            office_file.decrypt(decrypted_buffer)
        decrypted_buffer.seek(0)
        return Document(decrypted_buffer)

    def _extract_text(self, doc: Document) -> str:
        """Текст абзацев и таблиц, абзацы разделены пустой строкой."""
        parts = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        for table in doc.tables:
            for row in table.rows:
                row_cells = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]
                if row_cells:
                    parts.append(" | ".join(row_cells))

        return "\n\n".join(parts)

    def _extract_sections(self, doc: Document) -> list[Section]:
        """Заголовки (Heading 1-3 или «Заголовок») — границы секций.
        Без заголовков структуры нет — чанкинг пойдёт по предложениям."""
        sections: list[Section] = []
        has_heading = False
        current_title: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_lines
            body = "\n\n".join(current_lines).strip()
            if body or current_title is not None:
                sections.append(Section(title=current_title, text=body))
            current_lines = []

        for para in doc.paragraphs:
            text = para.text.strip()
            style = (para.style.name or "") if para.style is not None else ""
            if _HEADING_STYLE.match(style) and text:
                has_heading = True
                flush()
                current_title = text
            elif text:
                current_lines.append(text)
        if not has_heading:
            return []
        flush()
        return sections
