from pathlib import Path

from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.pdfpage import PDFPage

from .base import BaseParser, ParsedDocument

# OCR imports — only used if pdfminer returns too little text
try:
    import pytesseract
    from pdf2image import convert_from_path

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# If a page yields fewer characters than this threshold on average,
# we treat the PDF as scanned and fall back to OCR.
_CHARS_PER_PAGE_THRESHOLD = 100


class PdfParser(BaseParser):
    """
    Parses PDF files to plain text.

    Strategy:
    1. Try pdfminer for text-based PDFs (fast, accurate).
    2. Count characters per page. If below threshold → scanned PDF.
    3. Fall back to pytesseract OCR with Russian + English language packs.
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
        """Count PDF pages without extracting text."""
        with open(path, "rb") as f:
            return sum(1 for _ in PDFPage.get_pages(f))

    def _is_scanned(self, text: str, page_count: int) -> bool:
        """Return True if the PDF is likely a scanned image."""
        if page_count == 0:
            return False
        chars_per_page = len(text) / page_count
        return chars_per_page < _CHARS_PER_PAGE_THRESHOLD

    def _ocr_extract(self, path: Path) -> str:
        """
        Convert each PDF page to an image and run Tesseract OCR.
        Uses Russian + English language packs simultaneously.
        """
        if not OCR_AVAILABLE:
            raise EnvironmentError(
                "This PDF appears to be scanned, but pytesseract or pdf2image "
                "is not installed. Run: pip install pytesseract pdf2image\n"
                "Also install Tesseract OCR from your OS package manager."
            )

        # Convert PDF pages to PIL images at 300 DPI for good OCR accuracy
        images = convert_from_path(str(path), dpi=300)

        page_texts = []
        for image in images:
            page_text = pytesseract.image_to_string(
                image,
                lang="rus+eng",  # simultaneous Russian + English recognition
            )
            page_texts.append(page_text)

        return "\n\n".join(page_texts)
