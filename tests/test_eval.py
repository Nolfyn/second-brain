"""Тесты eval harness: оценка кейсов, метрики, целостность фикстур."""

import json
from pathlib import Path

import pytest

from tests.eval_retrieval import (
    FIXTURE_CORPUS,
    FIXTURE_QUERIES,
    compute_metrics,
    evaluate_case,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResult:
    def __init__(self, file_path):
        self.file_path = file_path


def _results(*paths):
    return [FakeResult(p) for p in paths]


def test_case_passes_when_all_must_found():
    case = {"query": "q", "must_find": ["Notes.docx"]}
    report = evaluate_case(_results("C:/docs/Notes.docx"), case)
    assert report["status"] == "PASS"
    assert report["missing"] == []


def test_case_fails_when_must_missing():
    case = {"query": "q", "must_find": ["Notes.docx", "Games.docx"]}
    report = evaluate_case(_results("C:/docs/Notes.docx"), case)
    assert report["status"] == "FAIL"
    assert report["missing"] == ["Games.docx"]


def test_case_fails_on_forbidden_file():
    case = {"query": "q", "must_find": ["Notes.docx"], "must_not_find": ["Grammar"]}
    report = evaluate_case(
        _results("C:/docs/Notes.docx", "C:/docs/reference/Grammar Reference.docx"),
        case,
    )
    assert report["status"] == "FAIL"
    assert report["forbidden"] == ["Grammar"]


def test_mrr_uses_first_must_find_rank():
    case = {"query": "q", "must_find": ["Games.docx"]}
    report = evaluate_case(
        _results("a/Notes.docx", "b/Games.docx"), case
    )
    assert report["rr"] == pytest.approx(0.5)


def test_compute_metrics():
    cases = [
        {"status": "PASS", "rr": 1.0, "found_count": 3},
        {"status": "FAIL", "rr": 0.5, "found_count": 2},
        {"status": "FAIL", "rr": 0.0, "found_count": 0},
    ]
    m = compute_metrics(cases)
    assert m["pass_rate"] == pytest.approx(1 / 3)
    assert m["mrr"] == pytest.approx(0.5)
    assert m["empty_rate"] == pytest.approx(1 / 3)


def test_fixture_corpus_exists():
    assert FIXTURE_CORPUS.exists(), "запусти: python tests/make_fixtures.py --corpus"


def test_all_referenced_files_exist_in_corpus():
    cases = json.loads(FIXTURE_QUERIES.read_text(encoding="utf-8"))
    corpus_files = {p.name.lower() for p in FIXTURE_CORPUS.rglob("*") if p.is_file()}
    for case in cases:
        for key in ("must_find", "may_find", "must_not_find"):
            for name in case.get(key, []):
                assert any(
                    name.lower() in f for f in corpus_files
                ), f"{case['query']!r}: {name!r} нет в корпусе фикстур"


def test_indexer_accepts_injected_stores(tmp_path):
    from brain.index import Embedder, VectorStore, DocumentStore, Indexer

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.txt").write_text("Синтетическая заметка про яхтинг.", encoding="utf-8")

    store = DocumentStore(str(tmp_path / "eval.db"))
    indexer = Indexer(
        embedder=Embedder(),
        vector_store=VectorStore(str(tmp_path / "chroma")),
        document_store=store,
    )
    indexer.index_folder(docs)

    assert store.stats()["total_files"] == 1
