"""
Skinny retrieval eval: checks that known-good queries surface their expected
source files in Retriever.retrieve() results.

Usage:
    python -m tests.eval_retrieval  # uses tests/eval_queries.json
    python -m tests.eval_retrieval path/to/other.json

Query file format (JSON list):
[
  {
    "query": "What articles have I read from example.com?",
    "must_find": ["Notes.docx"],
    "may_find": ["Journal 2025.docx"]
  }
]
"must_find": every listed file must appear in results (by filename substring,
case-insensitive) or the case FAILS.
"may_find": tolerated extras -- reported, never failed.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from brain.index import Embedder, VectorStore, DocumentStore  # noqa: E402
from brain.search.retriever import Retriever  # noqa: E402

DEFAULT_QUERIES = Path(__file__).parent / "eval_queries.json"


def run(queries_path: Path) -> int:
    if not queries_path.exists():
        print(f"Query file not found: {queries_path}")
        print(
            "Copy tests/eval_queries.example.json to tests/eval_queries.json "
            "and fill in your own cases."
        )
        return 2

    cases = json.loads(queries_path.read_text(encoding="utf-8"))

    retriever = Retriever(
        embedder=Embedder(),
        vector_store=VectorStore(),
        document_store=DocumentStore(),
    )

    passed = failed = 0
    for i, case in enumerate(cases, 1):
        query = case["query"]
        must = case.get("must_find", [])
        may = case.get("may_find", [])

        results = retriever.retrieve(query)
        found_files = {
            r.file_path.replace("\\", "/").split("/")[-1].lower()
            for r in results
        }

        missing = [
            m for m in must if not any(m.lower() in f for f in found_files)
        ]
        extras = [
            f
            for f in found_files
            if not any(m.lower() in f for m in must)
            and not any(x.lower() in f for x in may)
        ]

        status = "PASS" if not missing else "FAIL"
        if missing:
            failed += 1
        else:
            passed += 1

        print(f"[{status}] {i:2}. {query[:70]}")
        if missing:
            print(f"         missing: {missing}")
        if extras:
            print(f"         extra files: {sorted(extras)[:6]}")

    print(f"\n{passed} passed, {failed} failed, {len(cases)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_QUERIES
    sys.exit(run(path))
