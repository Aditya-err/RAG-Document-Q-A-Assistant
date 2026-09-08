from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class IngestResponse(BaseModel):
    filename: str
    success: bool
    num_chunks: int = 0
    num_pages: int = 0
    error: Optional[str] = None

class AskRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]]
    api_key: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

class RetrievedChunkSchema(BaseModel):
    source: str
    document_name: str
    page_number: Optional[int] = None
    text: str
    score: float

class RetrievalTraceSchema(BaseModel):
    query: str
    query_embedding_dim: int
    query_embedding_norm: float
    top_k_requested: int
    num_candidates_in_store: int
    num_results_returned: int

class GenerationTraceSchema(BaseModel):
    system_prompt: str
    context_char_count: int
    num_history_turns: int
    model: str
    temperature: float
    max_tokens: int

class FullTraceSchema(BaseModel):
    query: str
    retrieval_trace: RetrievalTraceSchema
    generation_trace: Optional[GenerationTraceSchema] = None
    context_text: str
    
    # Timings
    time_query_embedding: float = 0.0
    time_retrieval: float = 0.0
    time_context_building: float = 0.0
    time_llm_generation: float = 0.0
    time_total: float = 0.0

class GenerationResultSchema(BaseModel):
    answer: str
    trace: GenerationTraceSchema

class AskResponse(BaseModel):
    answer: str
    sources: List[RetrievedChunkSchema]
    trace: FullTraceSchema
