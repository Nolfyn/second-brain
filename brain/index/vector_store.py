from dataclasses import dataclass, field

import chromadb

import brain.config as config

_COLLECTION_NAME = "second_brain"


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    file_path: str
    file_type: str
    chunk_index: int
    score: float | None  # None, пока результат не оценён (семантика/реранк)
    source: str = field(default="semantic")  # semantic|keyword|file|year|topic
    section: str | None = None  # заголовок секции, если чанк структурный


class VectorStore:
    """
    Wraps ChromaDB behind a small interface: add() and search().
    """

    def __init__(self, db_dir: str | None = None):
        self._client = chromadb.PersistentClient(
            path=db_dir or str(config.chroma_dir())
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        chunk_ids: list[str],
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Add or update chunks. Existing IDs are overwritten (upsert)."""
        if not chunk_ids:
            return
        self._collection.upsert(
            ids=chunk_ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    def delete_by_file(self, file_path: str) -> None:
        """Remove all chunks belonging to a given file (used on re-index)."""
        self._collection.delete(where={"file_path": file_path})

    def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """Return the top_k most similar chunks to the query vector."""
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        if not results["ids"][0]:
            return []

        search_results = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # ChromaDB returns cosine distance; convert to a similarity score (0-1)
            score = 1 - distance

            search_results.append(
                SearchResult(
                    chunk_id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    file_path=metadata["file_path"],
                    file_type=metadata["file_type"],
                    chunk_index=metadata["chunk_index"],
                    score=round(score, 4),
                    section=metadata.get("section"),
                )
            )

        return search_results

    def count(self) -> int:
        """Total number of chunks currently indexed."""
        return self._collection.count()
