"""
text_processor.py
------------------
STAGE 2 of the RAG pipeline: TEXT PROCESSING / CLEANING.

Responsibility: take raw extracted text (which, especially from PDFs, is
often messy — broken hyphenation, stray whitespace, repeated newlines,
control characters) and normalize it BEFORE chunking.

Design note: cleaning happens before chunking (not after) because chunk
boundaries are character-position based. If we clean after chunking, we'd
risk chunks with inconsistent whitespace and could accidentally merge/split
words at chunk edges.

We deliberately do LIGHT cleaning only. Aggressive cleaning (e.g. removing
all punctuation, lowercasing) would hurt retrieval quality and destroy
information the LLM needs to answer accurately.
"""

import re


def clean_text(text: str) -> str:
    """Light, safe normalization of extracted document text."""
    if not text:
        return ""

    # 1. Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Remove null bytes / non-printable control chars (common in bad
    #    PDF extractions) but keep normal whitespace (\n, \t).
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # 3. Fix hyphenated line breaks from PDFs, e.g. "informa-\ntion" -> "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # 4. Collapse 3+ blank lines into a single blank line (paragraph break).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Collapse runs of spaces/tabs (but not newlines) into a single space.
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 6. Strip trailing whitespace on each line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()
