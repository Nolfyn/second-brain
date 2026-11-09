"""Тесты DocumentStore: матчинг файлов, prune, корпусная частотная фильтрация."""

from brain.index.document_store import DocumentStore


def _store(tmp_path):
    return DocumentStore(str(tmp_path / "test.db"))


def _record(store, tmp_path, name, chunks=2):
    """Создаёт реальный файл и его записи в хранилище."""
    p = tmp_path / name
    p.write_text("file content", encoding="utf-8")
    store.record_file(p, p.suffix.lstrip(".") or "docx", chunks)
    store.save_chunks(
        str(p), [(i, f"text of chunk {i}") for i in range(chunks)]
    )
    return str(p)


def test_find_files_matching_by_substring(tmp_path):
    store = _store(tmp_path)
    _record(store, tmp_path, "Games.docx")
    _record(store, tmp_path, "Notes.docx")

    matches = store.find_files_matching("games")
    assert [m.lower() for m in matches] == [str(tmp_path / "Games.docx").lower()]


def test_get_chunks_for_paths_ordered(tmp_path):
    store = _store(tmp_path)
    _record(store, tmp_path, "b.docx", chunks=2)
    a = _record(store, tmp_path, "a.docx", chunks=3)

    chunks = store.get_chunks_for_paths([a])
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.file_path == a for c in chunks)


def test_prune_missing_removes_records(tmp_path):
    store = _store(tmp_path)
    _record(store, tmp_path, "exists.docx")
    gone = tmp_path / "gone.docx"
    gone.write_text("x", encoding="utf-8")
    _record(store, tmp_path, "gone.docx")
    gone.unlink()  # файл удалён с диска после индексации

    removed = store.prune_missing()
    assert removed == [str(gone)]
    assert store.stats()["total_files"] == 1
    assert store.find_files_matching("gone") == []


def test_delete_records_removes_file_and_chunks(tmp_path):
    store = _store(tmp_path)
    p = _record(store, tmp_path, "obsolete.docx", chunks=2)

    store.delete_records(p)
    assert store.stats()["total_files"] == 0
    assert store.get_chunks_for_paths([p]) == []


def test_is_corpus_common_by_document_frequency(tmp_path):
    store = _store(tmp_path)
    texts = [
        [(0, "music rock albums")],
        [(0, "music jazz notes")],
        [(0, "music and zydeco")],
        [(0, "nothing common here")],
    ]
    for i, chunks in enumerate(texts):
        p = tmp_path / f"f{i}.docx"
        p.write_text("x", encoding="utf-8")
        store.record_file(p, "docx", 1)
        store.save_chunks(str(p), chunks)

    assert store.is_corpus_common("music")
    assert not store.is_corpus_common("zydeco")


def test_is_corpus_common_ignores_tiny_corpora(tmp_path):
    store = _store(tmp_path)
    p = tmp_path / "only.docx"
    p.write_text("x", encoding="utf-8")
    store.record_file(p, "docx", 1)
    store.save_chunks(str(p), [(0, "guitar guitar guitar")])

    # один файл - порог min 3 не достигнут, слово не считается общим
    assert not store.is_corpus_common("guitar")
