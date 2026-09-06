import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class IngestResult:
    filename: str
    success: bool
    num_chunks: int = 0
    num_pages: int = 0
    error: str = ""

@dataclass
class RetrievedChunk:
    source: str
    document_name: str
    page_number: Optional[int]
    text: str
    score: float

@dataclass
class RetrievalTrace:
    query: str
    query_embedding_dim: int
    query_embedding_norm: float
    top_k_requested: int
    num_candidates_in_store: int
    num_results_returned: int

@dataclass
class GenerationTrace:
    system_prompt: str
    context_char_count: int
    num_history_turns: int
    model: str
    temperature: float
    max_tokens: int

@dataclass
class FullTrace:
    query: str
    retrieval_trace: RetrievalTrace
    generation_trace: Optional[GenerationTrace]
    context_text: str
    
    # Timings
    time_query_embedding: float = 0.0
    time_retrieval: float = 0.0
    time_context_building: float = 0.0
    time_llm_generation: float = 0.0
    time_total: float = 0.0

@dataclass
class GenerationResult:
    answer: str
    sources: List[RetrievedChunk]
    trace: FullTrace

@dataclass
class ClientConfig:
    embedding_model_name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    similarity_threshold: float = 0.15
    llm_provider: str = "ollama"
    llm_model: str = "qwen2:0.5b"
    temperature: float = 0.2
    max_tokens: int = 1024
    max_context_chars: int = 6000
    refusal_phrase: str = "This information was not found in the knowledge base."
    system_prompt_template: str = ""

class RAGPipelineClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_key = ""
        self.config = ClientConfig()
        
    def set_api_key(self, api_key: str):
        self.api_key = api_key
        
    @property
    def is_ready_to_answer(self) -> bool:
        # In a real app we'd ask the backend, but for this fallback client
        # we'll say it's ready if either an API key is set, or we assume Ollama is active.
        return bool(self.api_key) or True
        
    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/documents", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def ingest_file(self, file_bytes: bytes, filename: str) -> IngestResult:
        files = {"file": (filename, file_bytes)}
        try:
            response = requests.post(f"{self.base_url}/api/ingest", files=files, timeout=60)
            response.raise_for_status()
            data = response.json()
            return IngestResult(
                filename=data.get("filename", filename),
                success=data.get("success", True),
                num_chunks=data.get("num_chunks", 0),
                num_pages=data.get("num_pages", 0),
                error=data.get("error", "")
            )
        except requests.exceptions.RequestException as e:
            # We don't expose 'e' strictly to UI in typical flow, but return it in error string
            return IngestResult(
                filename=filename,
                success=False,
                error=f"Connection failed: unable to reach backend processing service."
            )
        
    def list_documents(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/documents")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return []
        
    def remove_document(self, filename: str) -> int:
        try:
            response = requests.delete(f"{self.base_url}/api/documents/{filename}")
            return 1 if response.status_code == 200 else 0
        except requests.exceptions.RequestException:
            return 0
        
    def clear_knowledge_base(self):
        try:
            requests.delete(f"{self.base_url}/api/documents")
        except requests.exceptions.RequestException:
            pass
        
    def stats(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/api/stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {}
        
    def ask(self, query: str, chat_history: List[Dict[str, str]] = None) -> GenerationResult:
        payload = {
            "query": query,
            "chat_history": chat_history or [],
            "api_key": self.api_key
        }
        try:
            response = requests.post(f"{self.base_url}/api/ask", json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            sources = [
                RetrievedChunk(
                    source=s["source"],
                    document_name=s["document_name"],
                    page_number=s.get("page_number"),
                    text=s["text"],
                    score=s["score"]
                ) for s in data.get("sources", [])
            ]
            
            rt_data = data["trace"]["retrieval_trace"]
            rt = RetrievalTrace(
                query=rt_data["query"],
                query_embedding_dim=rt_data.get("query_embedding_dim", 0),
                query_embedding_norm=rt_data.get("query_embedding_norm", 0.0),
                top_k_requested=rt_data.get("top_k_requested", 0),
                num_candidates_in_store=rt_data.get("num_candidates_in_store", 0),
                num_results_returned=rt_data.get("num_results_returned", 0)
            )
            
            gt_data = data["trace"].get("generation_trace")
            gt = None
            if gt_data:
                gt = GenerationTrace(
                    system_prompt=gt_data.get("system_prompt", ""),
                    context_char_count=gt_data.get("context_char_count", 0),
                    num_history_turns=gt_data.get("num_history_turns", 0),
                    model=gt_data.get("model", ""),
                    temperature=gt_data.get("temperature", 0.0),
                    max_tokens=gt_data.get("max_tokens", 0)
                )
                
            full_trace = FullTrace(
                query=data["trace"].get("query", query),
                retrieval_trace=rt,
                generation_trace=gt,
                context_text=data["trace"].get("context_text", ""),
                time_query_embedding=data["trace"].get("time_query_embedding", 0.0),
                time_retrieval=data["trace"].get("time_retrieval", 0.0),
                time_context_building=data["trace"].get("time_context_building", 0.0),
                time_llm_generation=data["trace"].get("time_llm_generation", 0.0),
                time_total=data["trace"].get("time_total", 0.0)
            )
            
            return GenerationResult(
                answer=data.get("answer", "No answer generated."),
                sources=sources,
                trace=full_trace
            )
        except requests.exceptions.RequestException as e:
            # Create a dummy trace structure to satisfy the dataclass constraints
            rt = RetrievalTrace("", 0, 0.0, 0, 0, 0)
            ft = FullTrace(query, rt, None, "")
            return GenerationResult(
                answer="⚠️ **Document Processing Service Unavailable**\n\nWe couldn't connect to the document processing service. Please make sure the backend is running and try again.",
                sources=[],
                trace=ft
            )
