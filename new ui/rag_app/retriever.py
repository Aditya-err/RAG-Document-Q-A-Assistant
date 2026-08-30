"""
retriever.py
------------
STAGE 5 (retrieval half) of the RAG pipeline.

Responsibility: glue EmbeddingModel + VectorStore together into a single
`retrieve(query, top_k)` call, and capture enough intermediate data (the
query vector's shape/norm, raw scores) to support the RAG Trace/Debug view
requested for later.

Kept as its own module (rather than folded into vector_store.py) because
conceptually "retrieval" is a pipeline step that could later grow extra
logic — e.g. re-ranking, hybrid keyword+vector search, metadata filters —
without the vector store itself needing to change.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from embeddings import EmbeddingModel
from vector_store import VectorStore, RetrievedChunk


@dataclass
class RetrievalTrace:
    """Debug info about one retrieval call, for the future RAG Trace view."""
    query: str
    query_embedding_dim: int
    query_embedding_norm: float
    top_k_requested: int
    num_candidates_in_store: int
    num_results_returned: int


@dataclass
class RetrievalResult:
    results: List[RetrievedChunk]
    trace: RetrievalTrace


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self, query: str, top_k: int, min_score: float = 0.0
    ) -> RetrievalResult:
        query_vec: np.ndarray = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_vec, top_k=top_k, min_score=min_score)

        trace = RetrievalTrace(
            query=query,
            query_embedding_dim=int(query_vec.shape[0]),
            query_embedding_norm=float(np.linalg.norm(query_vec)),
            top_k_requested=top_k,
            num_candidates_in_store=self.vector_store.stats()["total_chunks"],
            num_results_returned=len(results),
        )
        return RetrievalResult(results=results, trace=trace)
