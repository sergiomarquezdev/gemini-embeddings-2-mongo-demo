"""Chunking helpers for text, PDF, audio, and video.

Each `chunk_*` function returns a list of dataclass instances with a uniform shape:
  - chunk_index (0-based)
  - n_total (total chunks for this source)
  - chunk-specific fields (text, page_start/end, time_start/end, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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
