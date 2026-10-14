"""Тесты сохранения секций в хранилище и индексатора с фейковым эмбеддером."""

import sqlite3

from brain.index import DocumentStore, VectorStore, Indexer


class FakeEmbedder:
    """Векторы-заглушки — индексация без загрузки модели."""

    def embed(self, texts):
        return [[0.0] * 384 for _ in texts]


def _make_indexer(tmp_path):
    return Indexer(
        embedder=FakeEmbedder(),
        vector_store=VectorStore(str(tmp_path / "chroma")),
        document_store=DocumentStore(str(tmp_path / "t.db")),
    )


def test_save_chunks_with_sections_roundtrip(tmp_path):
    store = DocumentStore(str(tmp_path / "t.db"))
    store.save_chunks(
        "f.docx",
        [(0, "первая часть", "Вступление"), (1, "вторая часть", "Выводы")],
    )

    chunks = store.get_chunks_for_paths(["f.docx"])
    assert [c.section for c in chunks] == ["Вступление", "Выводы"]


def test_save_chunks_without_sections_still_works(tmp_path):
    store = DocumentStore(str(tmp_path / "t.db"))
    store.save_chunks("f.docx", [(0, "текст")])

    chunks = store.get_chunks_for_paths(["f.docx"])
    assert chunks[0].section is None


def test_migration_adds_section_column(tmp_path):
    """Старая БД без колонки section открывается и мигрирует."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE files (file_path VARCHAR PRIMARY KEY, file_type "
        "VARCHAR NOT NULL, checksum VARCHAR NOT NULL, chunk_count INTEGER, "
        "indexed_at DATETIME)"
    )
    conn.execute(
        "CREATE TABLE chunks (chunk_id VARCHAR PRIMARY KEY, file_path "
        "VARCHAR NOT NULL, chunk_index INTEGER NOT NULL, raw_text TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO chunks VALUES ('f.docx::0', 'f.docx', 0, 'текст')"
    )
    conn.commit()
    conn.close()

    store = DocumentStore(str(db))
    chunks = store.get_chunks_for_paths(["f.docx"])
    assert chunks[0].raw_text == "текст"

    # после миграции можно писать секции
    store.save_chunks("f.docx", [(0, "текст", "Дата")])
    assert store.get_chunks_for_paths(["f.docx"])[0].section == "Дата"


def test_indexer_passes_docx_sections_to_store(tmp_path):
    from docx import Document

    docs = tmp_path / "docs"
    docs.mkdir()
    doc = Document()
    doc.add_heading("Начало", level=1)
    doc.add_paragraph("Текст начала. " * 200)  # тело больше порога чанка
    doc.add_heading("Конец", level=1)
    doc.add_paragraph("Текст конца. " * 200)
    doc.save(docs / "note.docx")

    indexer = _make_indexer(tmp_path)
    indexer.index_folder(docs)

    store = indexer._document_store
    chunks = store.get_chunks_for_paths([str(docs / "note.docx")])
    sections = {c.section for c in chunks}
    assert "Начало" in sections
    assert "Конец" in sections


def test_vector_metadata_contains_section(tmp_path):
    from docx import Document

    docs = tmp_path / "docs"
    docs.mkdir()
    doc = Document()
    doc.add_heading("Часть", level=1)
    doc.add_paragraph("Текст части.")
    doc.save(docs / "note.docx")

    indexer = _make_indexer(tmp_path)
    indexer.index_folder(docs)

    results = indexer._vector_store.search([0.0] * 384, top_k=5)
    assert results[0].section == "Часть"
