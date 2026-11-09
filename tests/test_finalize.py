"""Тесты финализации выдачи: лимиты путей и реранк."""

from brain.index import DocumentStore, VectorStore, SearchResult
from brain.search.retriever import Retriever


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] * 384 for _ in texts]


def _make_retriever(tmp_path, reranker=None):
    return Retriever(
        embedder=FakeEmbedder(),
        vector_store=VectorStore(str(tmp_path / "chroma")),
        document_store=DocumentStore(str(tmp_path / "t.db")),
        reranker=reranker,
    )


def _results(n):
    return [
        SearchResult(
            chunk_id=f"f.docx::{i}",
            text=f"текст {i}",
            file_path="C:/docs/f.docx",
            file_type="docx",
            chunk_index=i,
            score=None,
            source="year",
        )
        for i in range(n)
    ]


def test_default_cap_is_20(tmp_path):
    r = _make_retriever(tmp_path)
    assert len(r._finalize("q", _results(40))) == 20


def test_path_cap_keeps_up_to_60(tmp_path):
    r = _make_retriever(tmp_path)
    out = r._finalize("q", _results(40), max_results=60)
    assert len(out) == 40  # меньше лимита - не режем


def test_rerank_can_be_skipped(tmp_path):
    class FakeReranker:
        def __init__(self):
            self.calls = 0

        def rerank(self, query, results, top_n=15):
            self.calls += 1
            return results

    fake = FakeReranker()
    r = _make_retriever(tmp_path, reranker=fake)
    r._finalize("q", _results(40), max_results=60, rerank=False)
    assert fake.calls == 0

    r._finalize("q", _results(40))
    assert fake.calls == 1


def test_rerank_failure_falls_back_to_unranked(tmp_path):
    class BrokenReranker:
        def rerank(self, query, results, top_n=15):
            raise RuntimeError("модель недоступна")

    r = _make_retriever(tmp_path, reranker=BrokenReranker())
    out = r._finalize("q", _results(40))
    assert len(out) == 20


def test_rerank_floor_drops_confidently_irrelevant(tmp_path):
    class ScoredReranker:
        def rerank(self, query, results, top_n=15):
            for i, r in enumerate(results):
                r.score = 0.05 if i < 8 else 0.0
            return results

    r = _make_retriever(tmp_path, reranker=ScoredReranker())
    out = r._finalize("q", _results(12))
    assert len(out) == 8
    assert all((x.score or 0) >= 0.02 for x in out)


def test_pinned_results_survive_rerank_order(tmp_path):
    """Закрепленные результаты сохраняют порядок - реранкер
    работает только с хвостом."""

    class ReverseReranker:
        def rerank(self, query, results, top_n=15):
            return list(reversed(results))

    r = _make_retriever(tmp_path, reranker=ReverseReranker())
    out = r._finalize("q", _results(15), pinned=3)
    assert [x.chunk_index for x in out[:3]] == [0, 1, 2]
    assert [x.chunk_index for x in out[3:]] == list(range(14, 2, -1))
