"""Chunking helpers for text, PDF, audio, and video.

Each `chunk_*` function returns a list of dataclass instances with a uniform shape:
  - chunk_index (0-based)
  - n_total (total chunks for this source)
  - chunk-specific fields (text, page_start/end, time_start/end, etc.)
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

from pypdf import PdfReader, PdfWriter


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    n_total: int
    token_count: int


def chunk_text(
    text: str,
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int = 7000,
    overlap_tokens: int = 500,
) -> list[TextChunk]:
    """Split text into chunks of <= max_tokens using a binary-search-by-words approach.

    Args:
        text: source text.
        count_tokens: function (text -> int) — typically VertexClient.count_tokens.
        max_tokens: max tokens per chunk (default 7000, leaves margin vs 8192 hard limit).
        overlap_tokens: approx tokens to overlap between consecutive chunks.

    Returns:
        List of TextChunk. Empty list if text is empty/whitespace.
    """
    if not text.strip():
        return []
    if count_tokens(text) <= max_tokens:
        return [TextChunk(text=text, chunk_index=0, n_total=1, token_count=count_tokens(text))]

    words = text.split()
    chunks: list[TextChunk] = []
    start = 0
    # Word-to-token approximate ratio (refined per chunk)
    while start < len(words):
        end = _find_chunk_end(words, start, max_tokens, count_tokens)
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)
        chunks.append(
            TextChunk(
                text=chunk_text_str,
                chunk_index=len(chunks),
                n_total=-1,  # patched below
                token_count=count_tokens(chunk_text_str),
            )
        )
        if end >= len(words):
            break
        # Step forward leaving overlap (assume ~1.5 tokens per word; matches plan).
        overlap_word_estimate = max(1, int(overlap_tokens / 1.5))
        start = max(start + 1, end - overlap_word_estimate)

    n = len(chunks)
    return [TextChunk(text=c.text, chunk_index=c.chunk_index, n_total=n, token_count=c.token_count) for c in chunks]


def _find_chunk_end(words: list[str], start: int, max_tokens: int, count_tokens) -> int:
    """Binary search: largest end index such that count_tokens(words[start:end]) <= max_tokens."""
    lo, hi = start + 1, len(words)
    best = start + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = " ".join(words[start:mid])
        if count_tokens(candidate) <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


@dataclass
class PdfChunk:
    pdf_bytes: bytes
    page_start: int  # 1-based inclusive
    page_end: int    # 1-based inclusive
    chunk_index: int
    n_total: int


def chunk_pdf(pdf_bytes: bytes, *, max_pages: int = 4, overlap_pages: int = 1) -> list[PdfChunk]:
    """Split a PDF into chunks of <= max_pages with overlap_pages of overlap.

    Each chunk is a standalone PDF binary that Vertex can ingest directly.
    Page numbering is 1-based inclusive in returned metadata.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    n_pages = len(reader.pages)
    if n_pages <= 6:  # within hard limit, single chunk
        return [PdfChunk(pdf_bytes=pdf_bytes, page_start=1, page_end=n_pages, chunk_index=0, n_total=1)]

    chunks: list[PdfChunk] = []
    start = 0  # 0-based
    while start < n_pages:
        end = min(start + max_pages, n_pages)  # 0-based exclusive
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(
            PdfChunk(
                pdf_bytes=buf.getvalue(),
                page_start=start + 1,
                page_end=end,
                chunk_index=len(chunks),
                n_total=-1,
            )
        )
        if end >= n_pages:
            break
        start = end - overlap_pages

    n = len(chunks)
    return [PdfChunk(pdf_bytes=c.pdf_bytes, page_start=c.page_start, page_end=c.page_end, chunk_index=c.chunk_index, n_total=n) for c in chunks]
