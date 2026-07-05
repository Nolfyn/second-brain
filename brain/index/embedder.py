from sentence_transformers import SentenceTransformer

# Multilingual model: supports 50+ languages including Russian and English,
# and crucially supports CROSS-lingual search — a query in English can
# surface relevant Russian documents, and vice versa.
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """
    Thin wrapper around sentence-transformers.
    Loading the model is slow (~5-10s) so we load it once and reuse it.
    """

    def __init__(self):
        print(f"Loading embedding model '{_MODEL_NAME}'...")
        self._model = SentenceTransformer(_MODEL_NAME)
        print("Embedding model ready.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts. Returns one vector per input text,
        in the same order. Batching is handled internally by
        sentence-transformers for efficiency.
        """
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
        """Convenience method for embedding a single query string."""
        return self.embed([text])[0]
