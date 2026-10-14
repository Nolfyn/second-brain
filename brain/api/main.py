import json
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from brain.index import Embedder, VectorStore, DocumentStore
from brain.search.retriever import Retriever
from brain.search.reranker import Reranker
from brain.search.synthesiser import synthesise

log = logging.getLogger(__name__)

app = FastAPI(title="Second Brain API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

log.info("Loading shared instances...")
_embedder = Embedder()
_vector_store = VectorStore()
_document_store = DocumentStore()
_retriever = Retriever(
    embedder=_embedder,
    vector_store=_vector_store,
    document_store=_document_store,
    reranker=Reranker(),
)
log.info("Ready.")


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(request: AskRequest):
    """Пайплайн retrieve -> стриминг ответа LLM."""
    results = _retriever.retrieve(request.query)
    chunks = [
        {
            "file_path": r.file_path,
            "text": r.text,
            "score": r.score,
            "chunk_index": r.chunk_index,
            "source": r.source,
        }
        for r in results
    ]

    def stream():
        yield json.dumps({"type": "sources", "local": chunks}) + "\n"
        for token in synthesise(request.query, chunks):
            yield json.dumps({"type": "token", "content": token}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/stats")
def stats():
    return _document_store.stats()


@app.get("/debug/search")
def debug_search(q: str):
    """Обход LLM — показывает, что именно вернул ретривер."""
    results = _retriever.retrieve(q)
    return {
        "query": q,
        "result_count": len(results),
        "results": [
            {
                "file_path": r.file_path,
                "chunk_index": r.chunk_index,
                "score": r.score,
                "preview": r.text[:200],
            }
            for r in results
        ],
    }
