"""Переранжирование результатов мультиязычным cross-encoder'ом.

Добивает слабое место двухэтапного поиска: би-энкодер (MiniLM) грубый,
но быстрый; cross-encoder точный, но дорогой - поэтому он работает
только поверх уже отобранных кандидатов.
"""

import os
import threading

from brain.index.vector_store import SearchResult

_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"  # мультиязычный (RU/EN), скачивается с HF при первом запуске


class Reranker:
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or os.getenv(
            "RERANKER_MODEL", _DEFAULT_MODEL
        )
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        # Двойная проверка под локом: конкурентные первые запросы
        # не должны загрузить модель дважды
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder

                    print(
                        f"[RERANKER] loading cross-encoder "
                        f"'{self._model_name}'..."
                    )
                    self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, query: str, results: list[SearchResult], top_n: int = 15
    ) -> list[SearchResult]:
        """Сортирует кандидатов по релевантности запросу, обрезает до top_n."""
        if not results:
            return []
        model = self._load()
        scores = model.predict([(query, r.text) for r in results])
        scored = sorted(
            zip(results, scores), key=lambda p: p[1], reverse=True
        )
        out = []
        for r, s in scored[:top_n]:
            r.score = round(float(s), 4)
            out.append(r)
        return out
