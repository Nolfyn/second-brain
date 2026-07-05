import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import BaseParser, ParsedDocument
from .docx_parser import DocxParser


class DocParser(BaseParser):
    """
    Parses legacy .doc files by converting them to .docx with
    LibreOffice in headless mode, then delegating to DocxParser.

    The converted file lives in a temp directory and is deleted
    immediately after parsing — nothing is left on disk.
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

        # Override the metadata to reflect the original file
        return ParsedDocument(
            file_path=str(path),
            file_type="doc",
            raw_text=result.raw_text,
        )

    def _check_libreoffice(self) -> None:
        """Raise a clear error if LibreOffice is not found."""
        if (
            not shutil.which(self._libreoffice_path)
            and not Path(self._libreoffice_path).exists()
        ):
            raise EnvironmentError(
                f"LibreOffice not found at '{self._libreoffice_path}'.\n"
                "Install LibreOffice and set LIBREOFFICE_PATH in your .env.\n"
                "Download from: https://www.libreoffice.org/"
            )

    def _convert_to_docx(self, path: Path, output_dir: str) -> Path:
        """
        Run LibreOffice headless to convert .doc → .docx.
        Returns the path to the converted file.
        """
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
            timeout=60,  # 60s timeout — large files can be slow
        )

        if result.returncode != 0:
            raise ValueError(
                f"LibreOffice conversion failed for {path.name}.\n"
                f"stderr: {result.stderr}"
            )

        # LibreOffice names the output file <original_stem>.docx
        converted = Path(output_dir) / (path.stem + ".docx")
        if not converted.exists():
            raise ValueError(
                f"Conversion succeeded but output file not found: {converted}"
            )

        return converted
