from pathlib import Path

from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.pdfpage import PDFPage

from .base import BaseParser, ParsedDocument

# OCR-импорты нужны, только если pdfminer вернул слишком мало текста
try:
    import pytesseract
    from pdf2image import convert_from_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# Страница с текстом ниже этого порога в среднем — скан
_CHARS_PER_PAGE_THRESHOLD = 100


class PdfParser(BaseParser):
    """
    Парсинг PDF.

    1. pdfminer для текстовых PDF (быстро, точно).
    2. Если символов на страницу мало — считаем PDF сканом.
    3. Фолбэк: pytesseract OCR с русским и английским языками.
    """

    def parse(self, path: Path) -> ParsedDocument:
        self._check_exists(path)

        try:
            page_count = self._count_pages(path)
            text = pdfminer_extract(str(path))

            if self._is_scanned(text, page_count):
                text = self._ocr_extract(path)

            text = text.strip()
            if not text:
                raise ValueError("No text could be extracted from this PDF.")

            return ParsedDocument(
                file_path=str(path),
                file_type="pdf",
                raw_text=text,
                page_count=page_count,
            )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to parse {path.name}: {e}") from e

    def _count_pages(self, path: Path) -> int:
        """Число страниц без извлечения текста."""
        with open(path, "rb") as f:
            return sum(1 for _ in PDFPage.get_pages(f))

    def _is_scanned(self, text: str, page_count: int) -> bool:
        """True, если PDF похож на отсканированный образ."""
        if page_count == 0:
            return False
        chars_per_page = len(text) / page_count
        return chars_per_page < _CHARS_PER_PAGE_THRESHOLD

    def _ocr_extract(self, path: Path) -> str:
        """Каждая страница → изображение 300 DPI → Tesseract (rus+eng)."""
        if not OCR_AVAILABLE:
            raise EnvironmentError(
                "Этот PDF похож на скан, но pytesseract или pdf2image "
                "не установлены. Выполните: pip install pytesseract pdf2image\n"
                "Также установите Tesseract OCR из пакетного менеджера ОС."
            )

        images = convert_from_path(str(path), dpi=300)

        page_texts = []
        for image in images:
            page_text = pytesseract.image_to_string(
                image,
                lang="rus+eng",  # одновременное распознавание русского и английского
            )
            page_texts.append(page_text)

        return "\n\n".join(page_texts)
