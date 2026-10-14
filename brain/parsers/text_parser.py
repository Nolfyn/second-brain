"""Парсеры простых текстовых форматов: .txt, .csv, .html, .epub, .pptx.

Собственные минимальные лоадеры вместо langchain/unstructured —
без тяжёлых зависимостей и нативных бинарников.
"""

import csv
import io
from pathlib import Path

from .base import BaseParser, ParsedDocument

_SUPPORTED = {".txt", ".csv", ".html", ".epub", ".pptx"}


class TextParser(BaseParser):
    """Диспетчерит формат на конкретный лоадер."""

    def parse(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED:
            raise ValueError(
                f"TextParser does not support '{suffix}' files."
            )
        self._check_exists(path)

        text = _LOADERS[suffix](path)
        if not text.strip():
            raise ValueError("No text content extracted.")

        return ParsedDocument(
            file_path=str(path),
            file_type=suffix.lstrip("."),
            raw_text=text.strip(),
        )


def _load_txt(path: Path) -> str:
    return path.read_text(encoding=_detect_encoding(path))


def _load_csv(path: Path) -> str:
    # utf-8-sig прозрачно снимает BOM
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [", ".join(cell.strip() for cell in row) for row in csv.reader(f)]
    return "\n".join(r for r in rows if r.strip())


def _load_html(path: Path) -> str:
    import html2text

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True
    return converter.handle(
        path.read_text(encoding=_detect_encoding(path))
    )


def _load_epub(path: Path) -> str:
    import html2text
    import ebooklib
    from ebooklib import epub

    converter = html2text.HTML2Text()
    converter.ignore_links = True
    converter.ignore_images = True

    book = epub.read_epub(str(path))
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        body = converter.handle(item.get_content().decode("utf-8", "ignore"))
        if body.strip():
            parts.append(body)
    return "\n\n".join(parts)


def _load_pptx(path: Path) -> str:
    """Слайды: заголовки, текст фигур, заметки докладчика."""
    from pptx import Presentation

    prs = Presentation(str(path))
    parts = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        lines = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                line = " ".join(run.text for run in para.runs).strip()
                if line:
                    lines.append(line)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                lines.append(f"[Notes] {notes}")
        if lines:
            parts.append(f"[Slide {slide_num}]\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _detect_encoding(path: Path) -> str:
    """Кодировка файла: chardet, fallback UTF-8 → windows-1251."""
    try:
        import chardet

        raw = path.read_bytes()[:32768]
        detected = chardet.detect(raw)
        if detected.get("encoding") and (detected.get("confidence") or 0) >= 0.7:
            return detected["encoding"]
    except Exception:
        pass
    try:
        path.read_text(encoding="utf-8")
        return "utf-8"
    except (UnicodeDecodeError, Exception):
        return "windows-1251"


_LOADERS = {
    ".txt": _load_txt,
    ".csv": _load_csv,
    ".html": _load_html,
    ".epub": _load_epub,
    ".pptx": _load_pptx,
}
