from pathlib import Path

from .base import BaseParser
from .docx_parser import DocxParser
from .doc_parser import DocParser
from .pdf_parser import PdfParser
from .text_parser import TextParser


# Поддерживаемые расширения. Для нового формата: написать парсер
# и добавить расширение сюда — пайплайн (чанкинг, эмбеддинги, хранилище) не меняется.
_REGISTRY: dict[str, type[BaseParser]] = {
    # Свои парсеры для проблемных форматов: шифрование, OCR, legacy
    ".docx": DocxParser,
    ".doc": DocParser,
    ".pdf": PdfParser,
    # Простые текстовые форматы
    ".txt": TextParser,
    ".csv": TextParser,
    ".html": TextParser,
    ".epub": TextParser,
    ".pptx": TextParser,
}

SUPPORTED_EXTENSIONS = set(_REGISTRY.keys())


def get_parser(path: Path) -> BaseParser:
    """Парсер для файла. ValueError для неподдерживаемых типов."""
    suffix = path.suffix.lower()
    parser_class = _REGISTRY.get(suffix)

    if parser_class is None:
        raise ValueError(
            f"No parser available for '{suffix}' files. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return parser_class()
