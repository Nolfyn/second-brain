"""Тесты путей ретривера: keyword-путь расширяет попадания соседями."""

from brain.index import DocumentStore, VectorStore
from brain.search.retriever import Retriever


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 384 for _ in texts]

    def embed_one(self, text):
        return self.embed([text])[0]


def _retriever(tmp_path, store):
    return Retriever(
        embedder=FakeEmbedder(),
        vector_store=VectorStore(str(tmp_path / "chroma")),
        document_store=store,
    )


def test_keyword_path_expands_neighbours(tmp_path):
    store = DocumentStore(str(tmp_path / "t.db"))
    chunks = [(i, f"текст чанка {i}") for i in range(12)]
    chunks[6] = (6, "текст чанка 6 со hit")
    store.save_chunks("f.docx", chunks)

    r = _retriever(tmp_path, store)
    results = r.retrieve('расскажи про "hit"')

    assert len(results) == 7  # hit + ±3 соседа
    assert sorted(x.chunk_index for x in results) == [3, 4, 5, 6, 7, 8, 9]
    assert all(x.source == "keyword" for x in results)


def test_keyword_path_reads_as_contiguous_runs(tmp_path):
    """Расширенные попадания идут связным прогоном по chunk_index."""
    store = DocumentStore(str(tmp_path / "t.db"))
    chunks = [(i, f"текст {i}") for i in range(30)]
    chunks[6] = (6, "первый hit")
    chunks[20] = (20, "второй hit")
    store.save_chunks("f.docx", chunks)
    store.save_chunks("z.docx", [(0, "hit второй темы в другом файле")])

    r = _retriever(tmp_path, store)
    results = r.retrieve('что по "hit"')

    f_indices = [x.chunk_index for x in results if x.file_path == "f.docx"]
    assert f_indices[:7] == [3, 4, 5, 6, 7, 8, 9]
    assert f_indices[7:] == [17, 18, 19, 20, 21, 22, 23]


def test_keyword_path_scope_file_still_works(tmp_path):
    store = DocumentStore(str(tmp_path / "t.db"))
    a = tmp_path / "a.docx"
    a.write_text("x", encoding="utf-8")
    b = tmp_path / "b.docx"
    b.write_text("x", encoding="utf-8")
    store.record_file(a, "docx", 1)
    store.record_file(b, "docx", 2)
    store.save_chunks(str(a), [(0, "hit в файле a")])
    store.save_chunks(str(b), [(0, "hit в файле b"), (1, "продолжение b")])

    r = _retriever(tmp_path, store)
    results = r.retrieve('что в a.docx про "hit"')

    assert {x.file_path for x in results} == {str(a)}


def test_entity_boost_does_not_expand_neighbours(tmp_path):
    """Слабые entity-попадания не должны тащить соседей."""
    store = DocumentStore(str(tmp_path / "t.db"))
    chunks = [(i, f"текст {i}") for i in range(10)]
    chunks[3] = (3, "упоминание Qyzylorda вскользь")
    store.save_chunks("f.docx", chunks)

    r = _retriever(tmp_path, store)
    results = r.retrieve("Where is Qyzylorda?")

    assert len(results) == 1
    assert results[0].chunk_index == 3


def test_entity_dense_cluster_expanded_and_first(tmp_path):
    """Плотный кластер попаданий (заметка) расширяется соседями
    и ставится первым - связным прогоном."""
    store = DocumentStore(str(tmp_path / "t.db"))
    chunks = [(i, f"текст {i}") for i in range(30)]
    chunks[10] = (10, "Qyzylorda часть 1")
    chunks[11] = (11, "Qyzordyna продолжение")
    chunks[12] = (12, "Qyzylorda часть 3")
    chunks[25] = (25, "Qyzylorda вскользь в другой теме")
    store.save_chunks("f.docx", chunks)

    r = _retriever(tmp_path, store)
    results = r.retrieve("Where is Qyzylorda?")

    indices = [x.chunk_index for x in results]
    # кластер 10-12 расширен до 8-14 и стоит первым связным прогоном
    assert indices[:7] == [8, 9, 10, 11, 12, 13, 14]
    # одиночное упоминание - после, без расширения
    assert indices[7:] == [25]


def test_topic_path_full_coverage_for_small_sets(tmp_path):
    """Тематический запрос к маленькой папке отдаёт все чанки,
    а не семантическую выборку."""
    store = DocumentStore(str(tmp_path / "t.db"))
    f = tmp_path / "egypt" / "E.docx"
    f.parent.mkdir()
    f.write_text("x", encoding="utf-8")
    store.record_file(f, "docx", 5)
    store.save_chunks(str(f), [(i, f"день {i} подробности") for i in range(5)])

    r = _retriever(tmp_path, store)
    results = r.retrieve("Give an overview of my Egypt trip")

    assert len(results) == 5
    assert all(x.source == "topic" for x in results)


def test_expansion_junk_filtered(tmp_path, monkeypatch):
    """Мусорные варианты расширения отсекаются
    косинусной близостью к оригиналу."""

    class FakeEmb:
        def embed(self, texts):
            return [
                [1.0, 0.0]
                if ("Soviet" in t or "совет" in t.lower())
                else [0.0, 1.0]
                for t in texts
            ]

        def embed_one(self, text):
            return self.embed([text])[0]

    class JunkLLM:
        def chat(self, messages, *, stream=False, **kwargs):
            return "личные заметки\nзаметки"

    monkeypatch.setattr("brain.llm.get_llm", lambda: JunkLLM())
    r = Retriever(
        embedder=FakeEmb(),
        vector_store=VectorStore(str(tmp_path / "chroma")),
        document_store=DocumentStore(str(tmp_path / "t.db")),
    )
    queries = r._expand_query("economics of Soviet collapse")

    assert queries == ["economics of Soviet collapse"]
