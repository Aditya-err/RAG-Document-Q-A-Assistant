"""
app.py
------
STREAMLIT UI LAYER — intentionally minimal/functional right now.

Per the project brief: the RAG pipeline (rag_pipeline.py + everything it
wires together) is the real deliverable of this pass. This file exists only
so the pipeline is actually runnable and testable end-to-end. It will be
redesigned (dark dev-tool theme / Claude.ai-style theme, etc.) in the next
pass WITHOUT touching any file other than this one — that's the whole point
of keeping the backend separate.

Every piece of pipeline state lives in st.session_state so it survives
Streamlit's rerun-on-every-interaction model:
    - rag_pipeline: the RAGPipeline instance (holds the vector store)
    - chat_history: list of {"role", "content"} for the current conversation
    - last_trace: the AnswerTrace of the most recent answer (debug view)
"""

import streamlit as st

# pyrefly: ignore [missing-import]
from config import RAGConfig
# pyrefly: ignore [missing-import]
from embeddings import EmbeddingModel
# pyrefly: ignore [missing-import]
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Document Q&A", layout="wide")


# ---------------------------------------------------------------- caching
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model(model_name: str) -> EmbeddingModel:
    # Cached across reruns/sessions: loading a sentence-transformers model
    # is expensive (downloads + loads weights into memory), so we only want
    # to do it once per unique model_name for the whole server process.
    return EmbeddingModel(model_name)


def get_pipeline() -> RAGPipeline:
    if "rag_pipeline" not in st.session_state:
        config = RAGConfig()
        embedder = get_embedding_model(config.embedding_model_name)
        st.session_state.rag_pipeline = RAGPipeline(config, embedder)
        st.session_state.chat_history = []
        st.session_state.last_trace = None
    return st.session_state.rag_pipeline


pipeline = get_pipeline()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input("Anthropic API Key", type="password")
    if api_key:
        pipeline.set_api_key(api_key)

    st.subheader("Retrieval")
    top_k = st.slider("Top-K chunks", 1, 10, pipeline.config.top_k)
    pipeline.config.top_k = top_k

    st.divider()
    st.subheader("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Build / Update Index", disabled=not uploaded_files):
        for f in uploaded_files:
            result = pipeline.ingest_file(f.read(), f.name)
            if result.success:
                st.success(f"{result.filename}: {result.num_chunks} chunks indexed")
            else:
                st.error(f"{result.filename}: {result.error}")

    docs = pipeline.list_documents()
    if docs:
        st.caption(f"{len(docs)} document(s) indexed")
        for d in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(d)
            if col2.button("x", key=f"remove_{d}"):
                pipeline.remove_document(d)
                st.rerun()
    else:
        st.caption("No documents indexed yet.")

    stats = pipeline.stats()
    st.caption(f"Total chunks: {stats['total_chunks']}")

    if st.button("New Chat"):
        st.session_state.chat_history = []
        st.session_state.last_trace = None
        st.rerun()

# ------------------------------------------------------------------- main
st.title("RAG Document Q&A")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

query = st.chat_input("Ask a question about your documents...")

if query:
    if not pipeline.is_ready_to_answer:
        st.error("Enter your Anthropic API key in the sidebar first.")
    elif not pipeline.list_documents():
        st.error("Upload and index at least one document first.")
    else:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving + generating..."):
                result = pipeline.ask(query, chat_history=st.session_state.chat_history)
            st.write(result.answer)

            if result.sources:
                with st.expander(f"Sources ({len(result.sources)})"):
                    for s in result.sources:
                        st.markdown(f"**{s.source}** — similarity `{s.score:.3f}`")
                        st.caption(s.text[:300] + ("..." if len(s.text) > 300 else ""))

            with st.expander("RAG Trace / Debug"):
                st.json({
                    "query": result.trace.query,
                    "retrieval": vars(result.trace.retrieval_trace),
                    "generation": vars(result.trace.generation_trace) if result.trace.generation_trace else None,
                    "context_char_count": len(result.trace.context_text),
                })

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": result.answer})
        st.session_state.last_trace = result.trace
