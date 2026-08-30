"""
vector_store.py
----------------
STAGE 5 of the RAG pipeline: VECTOR STORE.

Responsibility: persist chunk embeddings + their metadata, support adding
new documents without destroying existing ones, support removing a single
document's chunks (for re-indexing), and support similarity search.

Implementation: FAISS `IndexFlatIP` (inner product). Combined with the
unit-normalization done in embeddings.py, inner product == cosine
similarity, so scores returned are directly interpretable as cosine
similarity in [-1, 1] (in practice ~[0, 1] for real text).

Why keep a parallel Python list of Chunk metadata instead of only relying
on FAISS? FAISS only stores vectors + integer ids — it has no concept of
"metadata". So we keep `self._chunks: List[Chunk]` where index i corresponds
to FAISS vector id i. This is the standard pattern for FAISS-backed RAG.

FAISS's IndexFlatIP does not support true "delete by id" cleanly across all
FAISS builds, so document removal is implemented as: filter out the removed
document's chunks/vectors in Python, then rebuild a fresh index from what's
left. For a portfolio-scale knowledge base (tens of thousands of chunks)
this is fast and, importantly, simple and correct.

Session persistence: `save()` / `load()` allow writing the index + metadata
to disk so a session's knowledge base can be reloaded later. The Streamlit
app itself uses st.session_state to keep this object alive for the duration
of a running session, which already satisfies "persist during the
application session" — save/load is an extra capability for durability.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: List[Chunk] = []

    # ---------------------------------------------------------------- add
    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Add new chunk vectors + metadata WITHOUT touching existing data.
        This is what makes indexing additive across multiple uploads."""
        if len(chunks) == 0:
            return
        assert embeddings.shape[0] == len(chunks), (
            "embeddings/chunks length mismatch"
        )
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    # ------------------------------------------------------------- search
    def search(
        self, query_embedding: np.ndarray, top_k: int, min_score: float = 0.0
    ) -> List[RetrievedChunk]:
        if self._index.ntotal == 0:
            return []
        top_k = min(top_k, self._index.ntotal)
        query = query_embedding.reshape(1, -1).astype("float32")
        scores, ids = self._index.search(query, top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            if score < min_score:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score)))
        return results

    # ------------------------------------------------------------ manage
    def list_documents(self) -> List[str]:
        seen = []
        for c in self._chunks:
            if c.document_name not in seen:
                seen.append(c.document_name)
        return seen

    def remove_document(self, document_name: str) -> int:
        """Remove all chunks belonging to `document_name` and rebuild the
        index from the remaining chunks. Returns number of chunks removed."""
        keep_chunks = [c for c in self._chunks if c.document_name != document_name]
        removed = len(self._chunks) - len(keep_chunks)
        if removed == 0:
            return 0

        new_index = faiss.IndexFlatIP(self.dimension)
        if keep_chunks:
            # Re-embed is NOT needed: FAISS lets us reconstruct stored
            # vectors directly instead of re-computing embeddings.
            all_vectors = self._index.reconstruct_n(0, self._index.ntotal)
            keep_mask = [c.document_name != document_name for c in self._chunks]
            kept_vectors = all_vectors[keep_mask]
            new_index.add(kept_vectors)

        self._index = new_index
        self._chunks = keep_chunks
        return removed

    def clear(self) -> None:
        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks = []

    def stats(self) -> dict:
        per_doc = {}
        for c in self._chunks:
            per_doc[c.document_name] = per_doc.get(c.document_name, 0) + 1
        return {
            "total_chunks": len(self._chunks),
            "total_documents": len(per_doc),
            "chunks_per_document": per_doc,
        }

    # -------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(Path(path) / "index.faiss"))
        with open(Path(path) / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    @classmethod
    def load(cls, path: str, dimension: int) -> "VectorStore":
        store = cls(dimension)
        index_path = Path(path) / "index.faiss"
        chunks_path = Path(path) / "chunks.pkl"
        if index_path.exists() and chunks_path.exists():
            store._index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                store._chunks = pickle.load(f)
        return store
