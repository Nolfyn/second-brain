from pathlib import Path

from .base import BaseParser
from .docx_parser import DocxParser
from .doc_parser import DocParser
from .pdf_parser import PdfParser
from .langchain_parser import LangChainParser


# File extensions this project supports.
# To add a new type: create a parser class and add its extension here.
# The rest of the pipeline (chunking, embedding, storage) never changes.
_REGISTRY: dict[str, type[BaseParser]] = {
    # Custom parsers (handle edge cases: encryption, OCR, legacy formats)
    ".docx": DocxParser,
    ".doc": DocParser,
    ".pdf": PdfParser,
    # LangChain-backed loaders (community ecosystem, format-aware)
    ".txt": LangChainParser,
    ".csv": LangChainParser,
    ".html": LangChainParser,
    ".epub": LangChainParser,
    ".pptx": LangChainParser,
}

SUPPORTED_EXTENSIONS = set(_REGISTRY.keys())


def get_parser(path: Path) -> BaseParser:
    """
    Return the correct parser instance for the given file path.
    Raises ValueError for unsupported file types.
    """
    suffix = path.suffix.lower()
    parser_class = _REGISTRY.get(suffix)

    if parser_class is None:
        raise ValueError(
            f"No parser available for '{suffix}' files. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return parser_class()
