import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from brain.chat_store import ChatStore
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
_chat_store = ChatStore()
log.info("Ready.")


class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask(request: AskRequest):
    """Пайплайн retrieve -> стриминг ответа LLM (без сохранения)."""
    chunks = _retrieve_chunks(request.query)

    def stream():
        yield json.dumps({"type": "sources", "local": chunks}) + "\n"
        for token in synthesise(request.query, chunks):
            yield json.dumps({"type": "token", "content": token}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/stats")
def stats():
    return _document_store.stats()


# ---------------------------------------------------------------------------
# Чат-сессии
# ---------------------------------------------------------------------------


@app.get("/sessions")
def list_sessions():
    return _chat_store.list_sessions()


@app.post("/sessions")
def create_session():
    return {"id": _chat_store.create_session()}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = _chat_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    _chat_store.delete_session(session_id)
    return {"deleted": session_id}


def _retrieve_chunks(query: str) -> list[dict]:
    results = _retriever.retrieve(query)
    return [
        {
            "file_path": r.file_path,
            "text": r.text,
            "score": r.score,
            "chunk_index": r.chunk_index,
            "source": r.source,
            "section": r.section,
        }
        for r in results
    ]


@app.post("/sessions/{session_id}/ask")
def ask_session(session_id: str, request: AskRequest):
    """Пайплайн /ask с сохранением переписки в сессию."""
    if _chat_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")

    def stream():
        _chat_store.add_message(session_id, "user", request.query)
        chunks = _retrieve_chunks(request.query)
        history = _chat_store.get_history(session_id, limit=10)[:-1]

        yield json.dumps({"type": "sources", "local": chunks}) + "\n"

        answer_parts = []
        for token in synthesise(request.query, chunks, history=history):
            answer_parts.append(token)
            yield json.dumps({"type": "token", "content": token}) + "\n"

        _chat_store.add_message(
            session_id,
            "assistant",
            "".join(answer_parts),
            sources=chunks,
        )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


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
