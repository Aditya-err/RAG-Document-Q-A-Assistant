"""
chunker.py
----------
STAGE 3 of the RAG pipeline: CHUNKING.

Responsibility: split cleaned document text into overlapping, meaningfully-
sized chunks, and attach the metadata every downstream stage depends on:
document name, page number (if any), a unique chunk_id, chunk_index, and a
human-readable "source" string used in citations.

ALGORITHM (recursive/hierarchical splitting, implemented from scratch so it
is easy to explain in an interview without pointing at a library):
  1. Try to split on paragraph breaks ("\n\n").
  2. If a paragraph is still bigger than chunk_size, split it on sentence
     boundaries (". ", "! ", "? ").
  3. If a single sentence is still bigger than chunk_size (rare — e.g. a
     giant table row), hard-split on chunk_size.
  4. Greedily pack these pieces into chunks up to chunk_size characters,
     carrying `chunk_overlap` characters of trailing context forward into
     the next chunk so retrieval doesn't lose meaning at chunk boundaries.

This runs independently per PDF "page" so page-number metadata is never
lost across a page boundary.
"""

import re
import uuid
from dataclasses import dataclass
from typing import List, Optional

from document_loader import LoadedDocument
from text_processor import clean_text


@dataclass
class Chunk:
    chunk_id: str
    document_name: str
    page_number: Optional[int]
    chunk_index: int
    text: str

    @property
    def source(self) -> str:
        """Human-readable citation string, e.g. 'report.pdf (p. 3)'."""
        if self.page_number is not None:
            return f"{self.document_name} (p. {self.page_number})"
        return self.document_name


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_into_pieces(text: str, chunk_size: int) -> List[str]:
    """Break text into paragraph -> sentence -> hard-cut pieces, each no
    larger than chunk_size where possible."""
    pieces: List[str] = []
    for para in _PARAGRAPH_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
            continue
        # Paragraph too big -> split into sentences.
        for sent in _SENTENCE_SPLIT.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= chunk_size:
                pieces.append(sent)
            else:
                # Sentence still too big -> hard cut.
                for i in range(0, len(sent), chunk_size):
                    pieces.append(sent[i:i + chunk_size])
    return pieces


def _pack_pieces(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Greedily pack small pieces into chunks up to chunk_size, carrying
    `overlap` trailing characters of the previous chunk into the next one."""
    chunks: List[str] = []
    current = ""

    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # Carry the tail of the finished chunk forward as overlap.
                tail = current[-overlap:] if overlap > 0 else ""
                current = f"{tail} {piece}".strip() if tail else piece
            else:
                # Single piece already exceeds chunk_size (shouldn't happen
                # after _split_into_pieces, but guard anyway).
                chunks.append(piece)
                current = ""

    if current:
        chunks.append(current)

    return chunks


def chunk_document(
    doc: LoadedDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """Turn a LoadedDocument into a flat list of Chunk objects, preserving
    page-level metadata."""
    chunks: List[Chunk] = []
    global_index = 0

    for page in doc.pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue

        pieces = _split_into_pieces(cleaned, chunk_size)
        packed = _pack_pieces(pieces, chunk_size, chunk_overlap)

        for text in packed:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_name=doc.filename,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    text=text,
                )
            )
            global_index += 1

    return chunks
