import io
import os
from pathlib import Path

import msoffcrypto
from docx import Document

from .base import BaseParser, ParsedDocument


class DocxParser(BaseParser):
    """
    Parses .docx files to plain text.
    Handles password-protected files transparently using the
    DOC_PASSWORD environment variable. Decryption is done
    entirely in memory — no plaintext is ever written to disk.
    """

    def __init__(self):
        self._password = os.getenv("DOC_PASSWORD", "")

    def parse(self, path: Path) -> ParsedDocument:
        self._check_exists(path)

        try:
            doc = self._open_document(path)
            text = self._extract_text(doc)
            return ParsedDocument(
                file_path=str(path),
                file_type="docx",
                raw_text=text,
            )
        except Exception as e:
            raise ValueError(f"Failed to parse {path.name}: {e}") from e

    def _open_document(self, path: Path) -> Document:
        """
        Try to open as a plain docx first.
        If that fails and we have a password, try decrypting.
        """
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
        """Decrypt an Office-encrypted file into an in-memory buffer."""
        with open(path, "rb") as f:
            office_file = msoffcrypto.OfficeFile(f)
            office_file.load_key(password=self._password)
            decrypted_buffer = io.BytesIO()
            office_file.decrypt(decrypted_buffer)
        decrypted_buffer.seek(0)
        return Document(decrypted_buffer)

    def _extract_text(self, doc: Document) -> str:
        """
        Pull text from paragraphs and tables.
        Preserves paragraph breaks; collapses runs of blank lines.
        """
        parts = []

        # Paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        # Tables (each cell as its own line)
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
