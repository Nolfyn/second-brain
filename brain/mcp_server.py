"""
MCP server exposing Second Brain's retrieval to MCP clients.
"""

import sys
import contextlib

with contextlib.redirect_stdout(sys.stderr):
    from dotenv import load_dotenv

    load_dotenv()

    from mcp.server.fastmcp import FastMCP

    from brain.index import Embedder, VectorStore, DocumentStore
    from brain.search.retriever import Retriever

    _retriever = Retriever(
        embedder=Embedder(),
        vector_store=VectorStore(),
        document_store=DocumentStore(),
    )
    _store = DocumentStore()

mcp = FastMCP("second-brain")

_MAX_CHUNK_CHARS = 1500
_MAX_FILE_CHUNKS = 60


@mcp.tool()
def search_notes(query: str) -> str:
    """
    Search the user's personal notes and documents. Handles English and
    Russian queries; content may be in either language. Returns the most
    relevant excerpts with their source files. Use get_note to read a
    full document found here.
    """
    with contextlib.redirect_stdout(sys.stderr):
        results = _retriever.retrieve(query)
    if not results:
        return "No relevant excerpts found for this query."
    parts = []
    for r in results:
        fname = r.file_path.replace("\\", "/").split("/")[-1]
        parts.append(
            f"[{fname} :: chunk {r.chunk_index}]\n{r.text[:_MAX_CHUNK_CHARS]}"
        )
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def get_note(filename: str) -> str:
    """
    Retrieve the full text of one indexed document by filename (or a
    distinctive part of it). Very large files are truncated.
    """
    with contextlib.redirect_stdout(sys.stderr):
        chunks = _store.get_chunks_for_file_matching(filename)
    if not chunks:
        return f"No indexed file matches '{filename}'."
    truncated = len(chunks) > _MAX_FILE_CHUNKS
    chunks = chunks[:_MAX_FILE_CHUNKS]
    body = "\n\n".join(c.raw_text for c in chunks)
    header = f"File: {chunks[0].file_path}" + (
        " (truncated)" if truncated else ""
    )
    return f"{header}\n\n{body}"


if __name__ == "__main__":
    mcp.run()
