"""
context_builder.py
-------------------
STAGE 6 of the RAG pipeline: AUGMENTATION (context construction).

Responsibility: turn a list of RetrievedChunk objects into a single string
that gets inserted into the LLM prompt, with each chunk clearly tagged by
its source so the model can cite it — and so a human reviewer (RAG Trace
view) can see exactly what the LLM was shown.

Also responsible for respecting `max_context_chars` so a query that matches
many long chunks can't blow up the prompt size / cost.
"""

from dataclasses import dataclass
from typing import List

from vector_store import RetrievedChunk


@dataclass
class BuiltContext:
    context_text: str
    used_chunks: List[RetrievedChunk]  # chunks actually included (after truncation)


def build_context(
    retrieved: List[RetrievedChunk], max_context_chars: int = 6000
) -> BuiltContext:
    parts: List[str] = []
    used: List[RetrievedChunk] = []
    running_len = 0

    for i, item in enumerate(retrieved, start=1):
        block = (
            f"[Source {i}: {item.chunk.source} | similarity={item.score:.3f}]\n"
            f"{item.chunk.text}\n"
        )
        if running_len + len(block) > max_context_chars and used:
            # Stop adding more chunks once we'd exceed the cap — but always
            # include at least one chunk even if it alone exceeds the cap.
            break
        parts.append(block)
        used.append(item)
        running_len += len(block)

    context_text = "\n---\n".join(parts) if parts else ""
    return BuiltContext(context_text=context_text, used_chunks=used)
