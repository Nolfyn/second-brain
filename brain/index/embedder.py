from sentence_transformers import SentenceTransformer

# Мультиязычная модель: 50+ языков включая русский и английский,
# кросс-языковой поиск (RU-запрос находит EN-документы и наоборот)
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """Тонкая обертка над sentence-transformers. Модель грузится один раз."""

    def __init__(self):
        print(f"Loading embedding model '{_MODEL_NAME}'...")
        self._model = SentenceTransformer(_MODEL_NAME)
        print("Embedding model ready.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Батч-эмбеддинг: один вектор на текст, порядок сохраняется."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Эмбеддинг одной строки (запрос)."""
        return self.embed([text])[0]
