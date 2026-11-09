"""Централизованные пути хранилищ.

По умолчанию - корень проекта: запуск из другой директории
не создаст второй пустой индекс. Переопределение - переменными окружения.
Значения читаются при каждом вызове, чтобы тесты могли менять env.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def chroma_dir() -> Path:
    return Path(os.getenv("BRAIN_CHROMA_DIR", PROJECT_ROOT / "chroma_db"))


def sqlite_path() -> Path:
    return Path(os.getenv("BRAIN_SQLITE_PATH", PROJECT_ROOT / "second_brain.db"))


def brainignore_path() -> Path:
    override = os.getenv("BRAIN_IGNORE_FILE")
    if override:
        return Path(override)
    cwd_file = Path(os.getcwd()) / ".brainignore"
    if cwd_file.exists():
        return cwd_file
    return PROJECT_ROOT / ".brainignore"
