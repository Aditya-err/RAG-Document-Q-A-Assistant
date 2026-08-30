"""
document_loader.py
-------------------
STAGE 1 of the RAG pipeline: DOCUMENT INGESTION.

Responsibility: turn a raw uploaded file (PDF / TXT / MD) into a
framework-agnostic `LoadedDocument` object containing plain text, split by
page where the format supports pages (PDF) and as a single page otherwise
(TXT/MD).

This module knows NOTHING about chunking, embeddings, or Streamlit. It only
extracts text + metadata. That separation is what lets us add a new file
type (e.g. DOCX) later by adding one function here, without touching any
other stage of the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import io

from pypdf import PdfReader


@dataclass
class PageContent:
    """One 'page' of extracted text. For TXT/MD there is only ever one page
    (page_number=None) since those formats have no page concept."""
    page_number: Optional[int]
    text: str


@dataclass
class LoadedDocument:
    """The result of ingesting one uploaded file."""
    filename: str
    doc_type: str  # "pdf" | "txt" | "md"
    pages: List[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


class UnsupportedFileTypeError(Exception):
    pass


def _load_pdf(file_bytes: bytes, filename: str) -> LoadedDocument:
    """Extract text page-by-page from a PDF so we can preserve page numbers
    in downstream chunk metadata (needed for citations later)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(PageContent(page_number=i, text=text))
    return LoadedDocument(filename=filename, doc_type="pdf", pages=pages)


def _load_txt(file_bytes: bytes, filename: str) -> LoadedDocument:
    text = file_bytes.decode("utf-8", errors="ignore")
    return LoadedDocument(
        filename=filename, doc_type="txt",
        pages=[PageContent(page_number=None, text=text)]
    )


def _load_md(file_bytes: bytes, filename: str) -> LoadedDocument:
    # We keep markdown syntax intact rather than stripping it. Headings like
    # "## Installation" are useful signal for chunk boundaries later, and
    # stripping them would lose structure for very little benefit.
    text = file_bytes.decode("utf-8", errors="ignore")
    return LoadedDocument(
        filename=filename, doc_type="md",
        pages=[PageContent(page_number=None, text=text)]
    )


def load_document(file_bytes: bytes, filename: str) -> LoadedDocument:
    """Dispatches to the right loader based on file extension.

    Args:
        file_bytes: raw bytes of the uploaded file.
        filename: original filename (used to detect extension + stored as
                  metadata for citations).
    """
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _load_pdf(file_bytes, filename)
    elif ext == ".txt":
        return _load_txt(file_bytes, filename)
    elif ext in (".md", ".markdown"):
        return _load_md(file_bytes, filename)
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported: .pdf, .txt, .md"
        )
