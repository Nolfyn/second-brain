"""Тесты централизованной конфигурации путей."""

from pathlib import Path

import brain.config as config


def test_paths_default_to_project_root(monkeypatch, tmp_path):
    for var in ("BRAIN_CHROMA_DIR", "BRAIN_SQLITE_PATH", "BRAIN_IGNORE_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # в CWD нет .brainignore

    assert config.chroma_dir() == config.PROJECT_ROOT / "chroma_db"
    assert config.sqlite_path() == config.PROJECT_ROOT / "second_brain.db"
    assert config.brainignore_path() == config.PROJECT_ROOT / ".brainignore"


def test_brainignore_prefers_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("BRAIN_IGNORE_FILE", raising=False)
    (tmp_path / ".brainignore").write_text("*x*", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert config.brainignore_path() == tmp_path / ".brainignore"


def test_paths_overridable_by_env(monkeypatch):
    monkeypatch.setenv("BRAIN_CHROMA_DIR", "D:/custom/chroma")
    monkeypatch.setenv("BRAIN_SQLITE_PATH", "D:/custom/brain.db")
    monkeypatch.setenv("BRAIN_IGNORE_FILE", "D:/custom/ignore.txt")

    assert config.chroma_dir() == Path("D:/custom/chroma")
    assert config.sqlite_path() == Path("D:/custom/brain.db")
    assert config.brainignore_path() == Path("D:/custom/ignore.txt")
