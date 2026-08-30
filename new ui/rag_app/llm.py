"""
llm.py
------
STAGE 7 of the RAG pipeline: GENERATION.

Responsibility: send (system prompt + retrieved context + chat history +
user query) to the LLM (Anthropic Claude) and return the grounded answer.

This module is intentionally the ONLY place that knows about the `anthropic`
SDK. If you wanted to swap in a different provider later, this is the only
file that would need to change — everything upstream (retriever, context
builder) is provider-agnostic.

We pass prior chat turns as real conversation history (not just flattened
into one string) so the model can naturally handle follow-up questions like
"what about the second point?".
"""

from dataclasses import dataclass
from typing import List, Dict

import anthropic

from config import RAGConfig


@dataclass
class GenerationTrace:
    """Debug info about one generation call, for the RAG Trace view."""
    system_prompt: str
    context_char_count: int
    num_history_turns: int
    model: str
    temperature: float
    max_tokens: int


@dataclass
class GenerationResult:
    answer: str
    trace: GenerationTrace


class LLMClient:
    def __init__(self, api_key: str, config: RAGConfig):
        self.api_key = api_key
        self.client = None
        if config.llm_provider == "claude":
            self.client = anthropic.Anthropic(
                api_key=api_key,
                default_headers={"Accept-Encoding": "identity"}
            )
        self.config = config

    def generate_answer(
        self,
        query: str,
        context_text: str,
        chat_history: List[Dict[str, str]],
    ) -> GenerationResult:
        """
        Args:
            query: the user's current question.
            context_text: the built context string from context_builder.py.
            chat_history: list of {"role": "user"|"assistant", "content": str}
                          from EARLIER turns (not including the current query).
        """
        system_prompt = self.config.system_prompt_template.format(
            refusal_phrase=self.config.refusal_phrase
        )

        if context_text.strip():
            user_turn = (
                f"CONTEXT:\n{context_text}\n\n"
                f"QUESTION:\n{query}"
            )
        else:
            # No chunks retrieved at all (empty knowledge base, or nothing
            # cleared the similarity threshold) -> tell the model plainly
            # so it gives the refusal phrase instead of guessing.
            user_turn = (
                "CONTEXT:\n(No relevant document chunks were retrieved for "
                "this question.)\n\n"
                f"QUESTION:\n{query}"
            )

        messages = list(chat_history) + [{"role": "user", "content": user_turn}]

        if self.config.llm_provider == "claude":
            response = self.client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                system=system_prompt,
                messages=messages,
            )
            answer_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
        elif self.config.llm_provider == "openai":
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            openai_messages = [{"role": "system", "content": system_prompt}] + messages
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.config.llm_model,
                    "messages": openai_messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=120
            )
            response.raise_for_status()
            answer_text = response.json()["choices"][0]["message"]["content"].strip()
        elif self.config.llm_provider == "gemini":
            import requests
            gemini_contents = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.llm_model}:generateContent?key={self.api_key}"
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": gemini_contents,
                    "generationConfig": {
                        "temperature": self.config.temperature,
                        "maxOutputTokens": self.config.max_tokens,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            answer_text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            import requests
            ollama_messages = [{"role": "system", "content": system_prompt}] + messages
            response = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.config.llm_model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_ctx": 4096
                    }
                },
                timeout=120,
            )
            response.raise_for_status()
            answer_text = response.json().get("message", {}).get("content", "").strip()

        trace = GenerationTrace(
            system_prompt=system_prompt,
            context_char_count=len(context_text),
            num_history_turns=len(chat_history),
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return GenerationResult(answer=answer_text, trace=trace)
