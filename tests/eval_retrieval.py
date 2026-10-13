"""Eval harness retrieval: известные запросы должны находить ожидаемые файлы.

Прогон без LLM (расширение запроса отключено):
    python -m tests.eval_retrieval
Полный прогон с LLM-расширением:
    python -m tests.eval_retrieval --with-expansion
Свой корпус и запросы:
    python -m tests.eval_retrieval --corpus <dir> --queries <file.json>

Формат запросов (JSON-список):
    {"query": "...", "must_find": ["File.docx"], "may_find": [...], "must_not_find": [...]}
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"
FIXTURE_QUERIES = Path(__file__).parent / "fixtures" / "eval_queries.json"


def _basename(file_path: str) -> str:
    return file_path.replace("\\", "/").split("/")[-1].lower()


def evaluate_case(results, case: dict) -> dict:
    """Сравнивает выдачу ретривера с ожиданиями кейса."""
    found = [_basename(r.file_path) for r in results]
    found_set = set(found)

    missing = [
        m
        for m in case.get("must_find", [])
        if not any(m.lower() in f for f in found_set)
    ]
    forbidden = [
        n
        for n in case.get("must_not_find", [])
        if any(n.lower() in f for f in found_set)
    ]

    rr = 0.0
    for m in case.get("must_find", []):
        for i, f in enumerate(found):
            if m.lower() in f:
                rr = max(rr, 1.0 / (i + 1))
                break

    return {
        "query": case["query"],
        "status": "PASS" if not missing and not forbidden else "FAIL",
        "missing": missing,
        "forbidden": forbidden,
        "found_count": len(found),
        "rr": rr,
    }


def compute_metrics(cases: list[dict]) -> dict:
    total = len(cases)
    return {
        "total": total,
        "passed": sum(1 for c in cases if c["status"] == "PASS"),
        "pass_rate": sum(1 for c in cases if c["status"] == "PASS") / total,
        "mrr": sum(c["rr"] for c in cases) / total,
        "empty_rate": sum(1 for c in cases if c["found_count"] == 0) / total,
    }


def build_retriever(workdir: Path, with_expansion: bool):
    """Индексирует корпус во временное хранилище и собирает ретривер."""
    from brain.index import Embedder, VectorStore, DocumentStore, Indexer
    from brain.search.retriever import Retriever

    embedder = Embedder()
    document_store = DocumentStore(str(workdir / "eval.db"))
    indexer = Indexer(
        embedder=embedder,
        vector_store=VectorStore(str(workdir / "chroma")),
        document_store=document_store,
    )
    indexer.index_folder(FIXTURE_CORPUS)

    if not with_expansion:
        # Детерминированный прогон: расширение запроса требует LLM
        Retriever._expand_query = lambda self, query: [query]

    return Retriever(
        embedder=embedder,
        vector_store=VectorStore(str(workdir / "chroma")),
        document_store=document_store,
    )


def run(
    queries_path: Path,
    corpus: Path,
    with_expansion: bool = False,
) -> tuple[list[dict], dict]:
    """Возвращает (отчёты по кейсам, сводные метрики)."""
    global FIXTURE_CORPUS
    FIXTURE_CORPUS = corpus

    cases = json.loads(queries_path.read_text(encoding="utf-8"))

    # ignore_cleanup_errors: Chroma держит mmap-дескрипторы, на Windows
    # удаление открытых файлов падает — временный каталог не критичен
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        # Свой .brainignore: изолирование от конфигурации проекта
        (workdir / ".brainignore").write_text(
            "# eval fixtures\n*grammar reference*\n", encoding="utf-8"
        )
        old_cwd = os.getcwd()
        os.chdir(workdir)
        try:
            retriever = build_retriever(workdir, with_expansion)
            reports = [
                evaluate_case(retriever.retrieve(c["query"]), c) for c in cases
            ]
        finally:
            os.chdir(old_cwd)

    return reports, compute_metrics(reports)


def main() -> int:
    # Консоль Windows по умолчанию cp1251 — фиксируем UTF-8 для вывода
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=FIXTURE_CORPUS)
    parser.add_argument("--queries", type=Path, default=FIXTURE_QUERIES)
    parser.add_argument(
        "--with-expansion",
        action="store_true",
        help="включить LLM-расширение запроса (требует запущенный Ollama)",
    )
    args = parser.parse_args()

    reports, metrics = run(args.queries, args.corpus, args.with_expansion)

    for r in reports:
        mark = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"[{mark}] {r['query']}")
        if r["missing"]:
            print(f"       не найдено: {r['missing']}")
        if r["forbidden"]:
            print(f"       нежелательно найдено: {r['forbidden']}")

    print()
    print(
        f"Итого: {metrics['passed']}/{metrics['total']} прошло, "
        f"MRR={metrics['mrr']:.3f}, пустых выдач={metrics['empty_rate']:.2%}"
    )
    return 0 if metrics["passed"] == metrics["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
