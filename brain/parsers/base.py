from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedDocument:
    """The output of any parser: clean text plus basic metadata."""

    file_path: str
    file_type: str  # "docx", "doc", "pdf"
    raw_text: str  # full extracted text, UTF-8
    page_count: int = 0  # 0 if not applicable


class BaseParser(ABC):
    """
    Strategy interface for all file parsers.
    Every parser implements exactly one method: parse().
    Adding support for a new file type = writing a new subclass.
    """

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """
        Extract clean UTF-8 text from the file at `path`.
        Raises FileNotFoundError if the file does not exist.
        Raises ValueError if the file cannot be parsed.
        """
        ...

    def _check_exists(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
