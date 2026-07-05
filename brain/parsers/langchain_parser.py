"""
LangChain document loader integration.

Covers file types the existing custom parsers don't handle:
  .txt   — plain text files
  .csv   — spreadsheet exports, data files
  .html  — locally saved web pages
  .epub  — e-books
  .pptx  — PowerPoint presentations

"""

from pathlib import Path

from .base import BaseParser, ParsedDocument

# Loader imports are deferred to _get_loader() to keep startup fast
# and give clear error messages if optional deps are missing.


class LangChainParser(BaseParser):
    """
    Wraps LangChain community document loaders.
    Supports: .txt, .csv, .html, .epub, .pptx
    """

    def parse(self, path: Path) -> ParsedDocument:
        self._check_exists(path)
        suffix = path.suffix.lower()

        try:
            loader = self._get_loader(path, suffix)
            documents = loader.load()  # list of LangChain Document objects
            text = self._merge(documents)

            if not text.strip():
                raise ValueError("No text content extracted.")

            return ParsedDocument(
                file_path=str(path),
                file_type=suffix.lstrip("."),
                raw_text=text,
            )
        except (ImportError, ModuleNotFoundError) as e:
            raise EnvironmentError(
                f"A required package for parsing {suffix} files is missing.\n"
                f"Run: pip install langchain-community unstructured ebooklib "
                f"html2text python-pptx\n"
                f"Original error: {e}"
            ) from e
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to parse {path.name}: {e}") from e

    def _get_loader(self, path: Path, suffix: str):
        """Return the appropriate LangChain loader for this file type."""
        path_str = str(path)

        if suffix == ".txt":
            from langchain_community.document_loaders import TextLoader

            # Detect encoding explicitly rather than relying on LangChain's
            # autodetect, which misidentifies short UTF-8 Cyrillic samples.
            encoding = self._detect_encoding(path)
            return TextLoader(path_str, encoding=encoding)

        elif suffix == ".csv":
            from langchain_community.document_loaders import CSVLoader

            return CSVLoader(
                path_str, encoding="utf-8-sig"
            )  # utf-8-sig strips BOM

        elif suffix == ".html":
            from langchain_community.document_loaders import BSHTMLLoader

            # BSHTMLLoader uses beautifulsoup4 -- simpler and more
            # portable than UnstructuredHTMLLoader which requires
            # native libmagic binaries (problematic on Windows).
            encoding = self._detect_encoding(path)
            return BSHTMLLoader(path_str, open_encoding=encoding)

        elif suffix == ".epub":
            from langchain_community.document_loaders import (
                UnstructuredEPubLoader,
            )

            return UnstructuredEPubLoader(path_str, mode="single")

        elif suffix == ".pptx":
            # Use python-pptx directly rather than LangChain's
            # UnstructuredPowerPointLoader, which hangs on Windows because
            # unstructured tries to invoke LibreOffice or other native tools.
            return _PptxLoader(path)

        else:
            raise ValueError(
                f"LangChainParser does not support '{suffix}' files."
            )

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        """
        Detect file encoding using chardet, with a sensible fallback order.
        Reads only the first 32KB -- enough for reliable detection without
        loading the whole file.
        Priority: detected encoding if confidence >= 0.7, otherwise UTF-8,
        then Windows-1251 (most common for Russian Windows files), then Latin-1.
        """
        try:
            import chardet

            raw = path.read_bytes()[:32768]
            detected = chardet.detect(raw)
            confidence = detected.get("confidence") or 0
            encoding = detected.get("encoding") or ""
            if encoding and confidence >= 0.7:
                return encoding
        except Exception:
            pass

        # If we couldn't detect reliably, try UTF-8 first -- by far the
        # most common encoding for modern files including Russian.
        try:
            path.read_text(encoding="utf-8")
            return "utf-8"
        except (UnicodeDecodeError, Exception):
            pass

        # Fallback for older Russian Windows files
        return "windows-1251"

    def _merge(self, documents: list) -> str:
        """
        Merge LangChain Document objects (or _PptxLoader pseudo-documents)
        into a single text string. Joins with double newlines so the
        sentence-aware chunker splits them correctly.
        """
        parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if content:
                source = doc.metadata.get("source") or ""
                page = doc.metadata.get("page_number") or doc.metadata.get(
                    "slide_number"
                )
                if page and ".pptx" in source.lower():
                    parts.append(f"[Slide {page}]\n{content}")
                else:
                    parts.append(content)

        return "\n\n".join(parts)


class _PseudoDocument:
    """Minimal duck-type of a LangChain Document for _merge() compatibility."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class _PptxLoader:
    """
    Direct python-pptx loader with a LangChain-compatible .load() interface.
    Extracts text from slide titles, body shapes, and speaker notes.
    Avoids unstructured / LibreOffice entirely -- no hangs on Windows.
    """

    def __init__(self, path: Path):
        self._path = path

    def load(self) -> list[_PseudoDocument]:
        from pptx import Presentation
        from pptx.util import Pt

        prs = Presentation(str(self._path))
        documents = []

        for slide_num, slide in enumerate(prs.slides, start=1):
            parts = []

            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                for para in shape.text_frame.paragraphs:
                    line = " ".join(run.text for run in para.runs).strip()
                    if line:
                        parts.append(line)

            # Include speaker notes if present
            if slide.has_notes_slide:
                notes_frame = slide.notes_slide.notes_text_frame
                notes_text = notes_frame.text.strip()
                if notes_text:
                    parts.append(f"[Notes] {notes_text}")

            if parts:
                documents.append(
                    _PseudoDocument(
                        page_content="\n".join(parts),
                        metadata={
                            "source": str(self._path),
                            "slide_number": slide_num,
                        },
                    )
                )

        return documents
