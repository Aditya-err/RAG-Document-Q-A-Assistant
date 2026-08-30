import sys
import os

# Add parent dir to path so we can import config, rag_pipeline, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from backend.schemas import (
    IngestResponse, AskRequest, AskResponse, RetrievedChunkSchema, 
    RetrievalTraceSchema, GenerationTraceSchema, FullTraceSchema
)
from config import RAGConfig
from embeddings import EmbeddingModel
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv(r"D:\project\RAG Document Q&A Assistant\files\rag_qa_assistant\rag_qa_assistant\.env")

app = FastAPI(title="Document Intelligence API")

# Global pipeline state
pipeline: RAGPipeline = None

@app.on_event("startup")
def startup_event():
    global pipeline
    print("Initializing ML models...")
    config = RAGConfig()
    embed_model = EmbeddingModel(config.embedding_model_name)
    pipeline = RAGPipeline(config=config, embedding_model=embed_model)
    if "ANTHROPIC_API_KEY" in os.environ:
        pipeline.set_api_key(os.environ["ANTHROPIC_API_KEY"])
    print("Pipeline ready.")

@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    contents = await file.read()
    result = pipeline.ingest_file(contents, file.filename)
    return IngestResponse(
        filename=result.filename,
        success=result.success,
        num_chunks=result.num_chunks,
        num_pages=result.num_pages,
        error=result.error
    )

@app.get("/api/documents", response_model=List[str])
def list_documents():
    return pipeline.list_documents()

@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    removed = pipeline.remove_document(filename)
    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document removed"}

@app.delete("/api/documents")
def clear_knowledge_base():
    pipeline.clear_knowledge_base()
    return {"message": "Knowledge base cleared"}

@app.get("/api/stats")
def get_stats():
    return pipeline.stats()

@app.post("/api/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    if req.api_key:
        pipeline.set_api_key(req.api_key)
        
    try:
        result = pipeline.ask(req.query, req.chat_history)
        
        # Map to schemas
        sources = [
            RetrievedChunkSchema(
                source=c.source,
                document_name=c.document_name,
                page_number=c.page_number,
                text=c.text,
                score=c.score
            ) for c in result.sources
        ]
        
        rt_data = result.trace.retrieval_trace
        retrieval_trace = RetrievalTraceSchema(
            query=rt_data.query,
            query_embedding_dim=rt_data.query_embedding_dim,
            query_embedding_norm=rt_data.query_embedding_norm,
            top_k_requested=rt_data.top_k_requested,
            num_candidates_in_store=rt_data.num_candidates_in_store,
            num_results_returned=rt_data.num_results_returned
        )
        
        gt = result.trace.generation_trace
        generation_trace = None
        if gt:
            generation_trace = GenerationTraceSchema(
                system_prompt=gt.system_prompt,
                context_char_count=gt.context_char_count,
                num_history_turns=gt.num_history_turns,
                model=gt.model,
                temperature=gt.temperature,
                max_tokens=gt.max_tokens
            )
            
        full_trace = FullTraceSchema(
            query=result.trace.query,
            retrieval_trace=retrieval_trace,
            generation_trace=generation_trace,
            context_text=result.trace.context_text
        )
        
        return AskResponse(
            answer=result.answer,
            sources=sources,
            trace=full_trace
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
