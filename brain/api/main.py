import json
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from brain.index import Embedder, VectorStore, DocumentStore
from brain.search.retriever import Retriever
from brain.search.synthesiser import synthesise

if os.getenv("AGENT_MODE", "classic") == "graph":
    from brain.search.agent_graph import GraphAgent as _AgentImpl
else:
    from brain.search.agent import Agent as _AgentImpl
from brain.search.web_search import web_search

app = FastAPI(title="Second Brain API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading shared instances...", file=sys.stderr)
_embedder = Embedder()
_vector_store = VectorStore()
_document_store = DocumentStore()
_retriever = Retriever(
    embedder=_embedder,
    vector_store=_vector_store,
    document_store=_document_store,
)
_agent = _AgentImpl(retriever=_retriever)
print("Ready.", file=sys.stderr)


class AskRequest(BaseModel):
    query: str
    top_k: int = 8
    web_search: bool = False


class AgentRequest(BaseModel):
    query: str
    web_search: bool = False


@app.post("/ask")
def ask(request: AskRequest):
    """Regular pipeline: retrieve -> stream LLM answer."""
    results = _retriever.retrieve(request.query)
    chunks = [
        {
            "file_path": r.file_path,
            "text": r.text,
            "score": r.score,
            "chunk_index": r.chunk_index,
        }
        for r in results
    ]
    web = web_search(request.query) if request.web_search else None

    def stream():
        # Sources first
        yield (
            json.dumps(
                {
                    "type": "sources",
                    "local": chunks,
                    "web": web or [],
                }
            )
            + "\n"
        )
        # Then streamed answer
        for token in synthesise(request.query, chunks, web):
            yield json.dumps({"type": "token", "content": token}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/agent")
def agent_ask(request: AgentRequest):
    """Agentic pipeline: model decides how many search calls to make."""

    def stream():
        for event in _agent.run(request.query, use_web=request.web_search):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/stats")
def stats():
    return _document_store.stats()


@app.get("/debug/search")
def debug_search(q: str):
    """Bypass LLM -- see exactly what the retriever returns."""
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
