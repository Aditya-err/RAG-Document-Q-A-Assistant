# RAG Document Q&A — Architecture

## Pipeline (each stage = one file, one responsibility)

```
Document Loader   (document_loader.py)   PDF/TXT/MD -> text + page metadata
      v
Text Processor    (text_processor.py)    clean/normalize text
      v
Chunker           (chunker.py)           paragraph/sentence-aware chunks + metadata
      v
Embedding Model    (embeddings.py)        sentence-transformers -> unit vectors
      v
Vector Store       (vector_store.py)      FAISS IndexFlatIP + metadata, add/remove/persist
      v
Retriever          (retriever.py)         query embed + similarity search + trace
      v
Context Builder    (context_builder.py)   retrieved chunks -> prompt context, capped size
      v
LLM                (llm.py)               Anthropic Claude call, grounded system prompt
      v
Answer + Sources    (rag_pipeline.py)      orchestrates all of the above, builds AnswerTrace
```

`rag_pipeline.py` is the single entry point (`RAGPipeline` class) that the
UI talks to. `app.py` is a deliberately minimal Streamlit UI whose only job
right now is to exercise every capability end-to-end — it will be redesigned
next without touching any other file.

## Why it's split this way

- Swap the embedding model → edit `config.py` only.
- Swap FAISS for another vector DB → rewrite `vector_store.py` only, same
  `add`/`search`/`remove_document` interface.
- Swap Claude for another LLM provider → rewrite `llm.py` only.
- Redesign the UI → rewrite `app.py` only.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter your Anthropic API key in the sidebar (never hardcoded), upload
PDF/TXT/MD files, click **Build / Update Index**, then ask questions in the
chat box. Each answer shows its sources (document, page, similarity score)
and a RAG Trace / Debug panel (query embedding stats, retrieval trace,
generation trace, context size) — this is the hook point for the future
Developer/RAG Trace mode.

## What's verified vs. what to verify on your machine

Everything except the actual sentence-transformers model download was
tested end-to-end in the build sandbox (which has no access to
huggingface.co): PDF page-aware extraction, text cleaning, chunking with
overlap, FAISS add/search/remove/persist with correct cosine scoring,
context truncation, and a full Streamlit boot with no errors. The
embedding model itself downloads from Hugging Face on first run on your
machine — that part just needs your normal internet connection, nothing
pipeline-specific.

## Known simplifications (fine for a portfolio project, worth naming in an interview)

- **Session-scoped KB with optional save/load**: the knowledge base lives
  in `st.session_state` per Streamlit session; `VectorStore.save()`/`load()`
  exist for on-disk persistence but aren't wired into the UI yet.
- **Document removal rebuilds the index** from FAISS's own stored vectors
  (no re-embedding needed) — fine at portfolio scale (thousands of chunks),
  would switch to `IndexIDMap` + a proper delete op at larger scale.
- **Chunking is character-based, not token-based** — simpler to explain,
  slightly less precise about LLM context budget than a tokenizer-aware
  splitter.
