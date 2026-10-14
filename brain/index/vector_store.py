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
    """ChromaDB за маленьким интерфейсом: add() и search()."""

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
        """Добавление или обновление чанков (upsert по id)."""
        if not chunk_ids:
            return
        self._collection.upsert(
            ids=chunk_ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    def delete_by_file(self, file_path: str) -> None:
        """Удаление всех чанков файла (переиндексация, prune)."""
        self._collection.delete(where={"file_path": file_path})

    def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """top_k самых похожих на вектор запроса чанков."""
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
            # ChromaDB возвращает косинусное расстояние; переводим в сходство
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
        """Число чанков в индексе."""
        return self._collection.count()
