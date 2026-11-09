"""Тесты расширения чанков соседями (контекст вокруг keyword-попаданий)."""

from brain.index import DocumentStore


def _seed(tmp_path):
    store = DocumentStore(str(tmp_path / "t.db"))
    chunks = [(i, f"текст чанка {i} обычный") for i in range(10)]
    chunks[5] = (5, "текст чанка 5 со hit внутри")
    store.save_chunks("f.docx", chunks)
    return store


def test_expand_neighbours_both_directions(tmp_path):
    store = _seed(tmp_path)
    hits = store.keyword_search("hit")
    expanded = store.expand_neighbours(hits, window=2)
    indices = sorted(c.chunk_index for c in expanded)
    assert indices == [3, 4, 5, 6, 7]


def test_expand_neighbours_clamped_at_file_edges(tmp_path):
    store = _seed(tmp_path)
    store.save_chunks("g.docx", [(0, "hit в начале"), (1, "следующий")])
    hits = store.keyword_search("hit")
    expanded = store.expand_neighbours(hits, window=2)
    g_indices = sorted(
        c.chunk_index for c in expanded if c.file_path == "g.docx"
    )
    assert g_indices == [0, 1]


def test_expand_neighbours_dedup_and_no_hits(tmp_path):
    store = _seed(tmp_path)
    assert store.expand_neighbours([], window=2) == []
    hits = store.keyword_search("текст")  # попадание во все чанки
    expanded = store.expand_neighbours(hits, window=1)
    assert len({(c.file_path, c.chunk_index) for c in expanded}) == len(
        expanded
    )
