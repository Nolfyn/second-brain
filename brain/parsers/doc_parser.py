import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import BaseParser, ParsedDocument
from .docx_parser import DocxParser


class DocParser(BaseParser):
    """
    Легаси .doc: конвертация в .docx через LibreOffice (headless),
    дальше парсинг делегируется DocxParser. Конвертированный файл находится
    во временной директории и удаляется сразу после парсинга.
    """

    def __init__(self):
        self._libreoffice_path = os.getenv("LIBREOFFICE_PATH", "soffice")
        self._docx_parser = DocxParser()

    def parse(self, path: Path) -> ParsedDocument:
        self._check_exists(path)
        self._check_libreoffice()

        with tempfile.TemporaryDirectory() as tmp_dir:
            converted_path = self._convert_to_docx(path, tmp_dir)
            result = self._docx_parser.parse(converted_path)

        # Метаданные относятся к исходному файлу, не к конверту
        return ParsedDocument(
            file_path=str(path),
            file_type="doc",
            raw_text=result.raw_text,
        )

    def _check_libreoffice(self) -> None:
        """Понятная ошибка, если LibreOffice не найден."""
        if (
            not shutil.which(self._libreoffice_path)
            and not Path(self._libreoffice_path).exists()
        ):
            raise EnvironmentError(
                f"LibreOffice не найден: '{self._libreoffice_path}'.\n"
                "Установите LibreOffice и задайте LIBREOFFICE_PATH в .env.\n"
                "Скачать: https://www.libreoffice.org/"
            )

    def _convert_to_docx(self, path: Path, output_dir: str) -> Path:
        """Конвертация .doc → .docx через headless LibreOffice."""
        cmd = [
            self._libreoffice_path,
            "--headless",
            "--convert-to",
            "docx",
            "--outdir",
            output_dir,
            str(path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # большие файлы конвертируются долго
        )

        if result.returncode != 0:
            raise ValueError(
                f"LibreOffice не смог сконвертировать {path.name}.\n"
                f"stderr: {result.stderr}"
            )

        # LibreOffice называет результат <имя_без_расширения>.docx
        converted = Path(output_dir) / (path.stem + ".docx")
        if not converted.exists():
            raise ValueError(
                f"Конвертация прошла, но файл не найден: {converted}"
            )

        return converted
