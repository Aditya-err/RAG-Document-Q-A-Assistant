"""
config.py
---------
Central configuration for the entire RAG pipeline.

WHY THIS FILE EXISTS (interview talking point):
Every RAG system has a handful of knobs that control quality/cost/speed
trade-offs: chunk size, overlap, top-k, which embedding model, which LLM,
temperature, etc. Hardcoding these inside pipeline logic makes the system
hard to tune and hard to reason about. By keeping them in one dataclass,
the UI layer (Streamlit) can expose sliders/dropdowns that mutate a single
object, and the backend never needs to change.

Nothing in this file talks to Streamlit, FAISS, sentence-transformers, or
Anthropic directly. It is pure configuration.
"""

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    # ---- Embedding settings ----
    # A small, fast, well-regarded sentence-embedding model. 384-dim output.
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ---- Chunking settings ----
    # Measured in characters (not tokens) to keep the splitter dependency-free.
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # ---- Retrieval settings ----
    top_k: int = 4
    # Cosine similarity floor. Chunks below this are dropped even if they
    # are in the top-k, so we don't force irrelevant context into the prompt.
    similarity_threshold: float = 0.15

    # ---- LLM settings ----
    llm_provider: str = "ollama"  # "claude" or "ollama"
    llm_model: str = "qwen2:0.5b"
    temperature: float = 0.2
    max_tokens: int = 1024

    # ---- Context construction ----
    # Hard cap on how many characters of retrieved text we stuff into the
    # prompt, regardless of top_k, to keep prompts bounded.
    max_context_chars: int = 6000

    # ---- Grounding behavior ----
    refusal_phrase: str = "This information was not found in the knowledge base."

    system_prompt_template: str = field(default_factory=lambda: (
        "You are a document question-answering assistant. You must answer "
        "ONLY using the CONTEXT provided below, which was retrieved from the "
        "user's uploaded documents. Do not use outside knowledge and do not "
        "guess or invent facts.\n\n"
        "Rules:\n"
        "1. If the answer is present in the context, answer clearly and "
        "concisely, and refer to which source(s) you used.\n"
        "2. If the context does not contain the answer, respond EXACTLY with: "
        "\"{refusal_phrase}\"\n"
        "3. Never fabricate document names, page numbers, or facts not "
        "present in the context.\n"
        "4. You may synthesize information across multiple chunks if they "
        "are all part of the provided context.\n"
    ))
