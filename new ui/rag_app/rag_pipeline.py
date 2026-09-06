"""
rag_pipeline.py
----------------
ORCHESTRATOR — wires together every stage:

    Document Loader -> Text Processor -> Chunker -> Embedding Model
    -> Vector Store -> Retriever -> Context Builder -> LLM -> Answer + Sources

This is the ONLY module the Streamlit UI (app.py) should import from for
actual RAG behavior. The UI should never call document_loader, chunker,
vector_store, etc. directly — it goes through RAGPipeline. That's what
lets the UI be redesigned later without touching pipeline internals.

Also assembles `AnswerTrace`, the full record of what happened for one
question — this is what a future "Developer / RAG Trace mode" panel in the
UI would render:

    User Query
       v
    Query Embedding
       v
    Similarity Retrieval
       v
    Top-K Chunks
       v
    Context Construction
       v
    LLM
       v
    Grounded Answer
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config import RAGConfig
from document_loader import load_document, UnsupportedFileTypeError
from text_processor import clean_text  # noqa: F401 (used inside chunker; re-exported for clarity)
from chunker import chunk_document, Chunk
from embeddings import EmbeddingModel
from vector_store import VectorStore, RetrievedChunk
from retriever import Retriever, RetrievalTrace
from context_builder import build_context
from llm import LLMClient, GenerationTrace


@dataclass
class IngestResult:
    filename: str
    success: bool
    num_chunks: int = 0
    num_pages: int = 0
    error: Optional[str] = None


@dataclass
class SourceInfo:
    """What the UI needs to render one citation."""
    source: str            # e.g. "handbook.pdf (p. 4)"
    document_name: str
    page_number: Optional[int]
    text: str
    score: float


@dataclass
class AnswerTrace:
    """Full record of one question -> answer cycle, for RAG Trace/Debug UI."""
    query: str
    retrieval_trace: RetrievalTrace
    generation_trace: Optional[GenerationTrace]
    context_text: str
    sources: List[SourceInfo]
    
    # Timings
    time_query_embedding: float = 0.0
    time_retrieval: float = 0.0
    time_context_building: float = 0.0
    time_llm_generation: float = 0.0
    time_total: float = 0.0


@dataclass
class AnswerResult:
    answer: str
    sources: List[SourceInfo]
    trace: AnswerTrace


class RAGPipeline:
    def __init__(self, config: RAGConfig, embedding_model: EmbeddingModel):
        self.config = config
        self.embedding_model = embedding_model
        self.vector_store = VectorStore(dimension=embedding_model.dimension)
        self.retriever = Retriever(embedding_model, self.vector_store)
        if config.llm_provider == "ollama":
            self._llm_client = LLMClient(api_key="", config=config)
        else:
            self._llm_client: Optional[LLMClient] = None

    # ------------------------------------------------------------- setup
    def set_api_key(self, api_key: str) -> None:
        self._llm_client = LLMClient(api_key=api_key, config=self.config)

    @property
    def is_ready_to_answer(self) -> bool:
        return self._llm_client is not None

    # ----------------------------------------------------------- ingest
    def ingest_file(self, file_bytes: bytes, filename: str) -> IngestResult:
        """Runs stages 1-4 for a single uploaded file and adds it to the
        vector store WITHOUT touching previously indexed documents."""
        try:
            doc = load_document(file_bytes, filename)
        except UnsupportedFileTypeError as e:
            return IngestResult(filename=filename, success=False, error=str(e))

        if doc.total_chars == 0:
            return IngestResult(
                filename=filename, success=False,
                error="No extractable text found (file may be empty, scanned, or image-only).",
            )

        chunks: List[Chunk] = chunk_document(
            doc,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if not chunks:
            return IngestResult(
                filename=filename, success=False,
                error="Text extracted but produced zero chunks.",
            )

        texts = [c.text for c in chunks]
        vectors = self.embedding_model.embed_texts(texts)
        self.vector_store.add(vectors, chunks)

        return IngestResult(
            filename=filename, success=True,
            num_chunks=len(chunks), num_pages=len(doc.pages),
        )

    # --------------------------------------------------------- kb manage
    def list_documents(self) -> List[str]:
        return self.vector_store.list_documents()

    def remove_document(self, filename: str) -> int:
        return self.vector_store.remove_document(filename)

    def clear_knowledge_base(self) -> None:
        self.vector_store.clear()

    def stats(self) -> dict:
        return self.vector_store.stats()

    # ------------------------------------------------------------- ask
    def ask(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        top_k: Optional[int] = None,
    ) -> AnswerResult:
        """Runs stages 5-8 for one user question.

        chat_history: prior turns as [{"role": "user"/"assistant", "content": str}, ...]
                      NOT including the current `query`.
        """
        import time
        t_start_total = time.perf_counter()
        
        if not self.is_ready_to_answer:
            raise RuntimeError("LLM client not configured — call set_api_key() first.")

        k = top_k if top_k is not None else self.config.top_k

        # The retrieval inside self.retriever.retrieve also does query embedding.
        # We can approximate by measuring the whole retrieve block for time_retrieval.
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(
            query, top_k=k, min_score=self.config.similarity_threshold
        )
        t_retrieval = time.perf_counter() - t0

        t0 = time.perf_counter()
        built = build_context(retrieval.results, max_context_chars=self.config.max_context_chars)

        final_context = built.context_text
        t_context_building = time.perf_counter() - t0

        t0 = time.perf_counter()
        generation = self._llm_client.generate_answer(
            query=query, context_text=final_context, chat_history=chat_history
        )
        t_llm_generation = time.perf_counter() - t0

        sources = [
            SourceInfo(
                source=item.chunk.source,
                document_name=item.chunk.document_name,
                page_number=item.chunk.page_number,
                text=item.chunk.text,
                score=item.score,
            )
            for item in built.used_chunks
        ]
        
        t_total = time.perf_counter() - t_start_total

        trace = AnswerTrace(
            query=query,
            retrieval_trace=retrieval.trace,
            generation_trace=generation.trace,
            context_text=built.context_text,
            sources=sources,
            time_query_embedding=0.0, # Part of retrieve
            time_retrieval=t_retrieval,
            time_context_building=t_context_building,
            time_llm_generation=t_llm_generation,
            time_total=t_total
        )

        return AnswerResult(answer=generation.answer, sources=sources, trace=trace)
