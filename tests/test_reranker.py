"""Тесты reranker'а (cross-encoder — мок, без загрузки модели)."""

from unittest.mock import MagicMock, patch

from brain.index.vector_store import SearchResult
from brain.search.reranker import Reranker


def _result(text, score=None):
    return SearchResult(
        chunk_id="f::0",
        text=text,
        file_path="f.docx",
        file_type="docx",
        chunk_index=0,
        score=score,
    )


def test_lazy_load_does_not_import_model():
    r = Reranker()
    assert r._model is None


def test_rerank_orders_by_model_scores():
    results = [_result("aaa"), _result("bbb"), _result("ccc")]
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    with patch.object(Reranker, "_load", return_value=mock_model):
        out = Reranker().rerank("q", results, top_n=2)

    assert [r.text for r in out] == ["bbb", "ccc"]
    assert out[0].score == 0.9


def test_rerank_empty_input():
    assert Reranker().rerank("q", []) == []
