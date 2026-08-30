"""
embeddings.py
-------------
STAGE 4 of the RAG pipeline: EMBEDDINGS.

Responsibility: convert text (chunks at index time, queries at retrieval
time) into fixed-size vectors using a local sentence-transformers model.

Deliberately framework-agnostic: this module has no Streamlit import. The
app layer decides whether/how to cache the loaded model (e.g. with
st.cache_resource) — this file just exposes a plain class.

We normalize embeddings to unit length so that a FAISS inner-product index
(IndexFlatIP) is equivalent to cosine similarity search. This keeps
vector_store.py simple: it doesn't need to know about normalization, it just
does inner-product search on whatever vectors it's given, as long as
embeddings.py guarantees they're pre-normalized.
"""

from typing import List
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12  # avoid divide-by-zero on empty text
        return vectors / norms

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of chunk texts. Returns shape (N, dim), unit-normalized."""
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vectors = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        return self._normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query. Returns shape (dim,), unit-normalized."""
        vector = self._model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        return self._normalize(vector)[0]
