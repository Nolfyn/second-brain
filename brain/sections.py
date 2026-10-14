"""Общие типы данных, используемые парсерами и индексатором."""

from dataclasses import dataclass


@dataclass
class Section:
    """Смысловая часть документа с необязательным заголовком."""

    title: str | None
    text: str
