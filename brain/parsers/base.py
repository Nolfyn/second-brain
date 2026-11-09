from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from brain.sections import Section


@dataclass
class ParsedDocument:
    """Результат парсера: чистый текст и базовые метаданные."""

    file_path: str
    file_type: str  # "docx", "doc", "pdf", ...
    raw_text: str  # полный текст, UTF-8
    page_count: int = 0  # 0, если неприменимо
    # Структурные секции (заголовки, даты). Пустой список - структуры нет,
    # чанкинг идёт по предложениям.
    sections: list[Section] = field(default_factory=list)


class BaseParser(ABC):
    """Интерфейс парсеров: ровно один метод parse()."""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """
        Извлекает чистый UTF-8 текст из файла.
        FileNotFoundError - файла нет. ValueError - распарсить не удалось.
        """
        ...

    def _check_exists(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
