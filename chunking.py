"""Chunking helpers for text, PDF, audio, and video.

Each `chunk_*` function returns a list of dataclass instances with a uniform shape:
  - chunk_index (0-based)
  - n_total (total chunks for this source)
  - chunk-specific fields (text, page_start/end, time_start/end, etc.)
"""
from __future__ import annotations

import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class AudioChunk:
    audio_bytes: bytes  # WAV format
    time_start: int     # seconds, 0-based
    time_end: int       # seconds, exclusive
    chunk_index: int
    n_total: int


def _audio_duration_seconds(data: bytes) -> float:
    """Get audio duration in seconds. Uses a temp file to avoid WAV pipe header issues."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", "-i", tmp_path],
            capture_output=True, check=True,
        )
        return float(p.stdout.decode().strip())
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _slice_audio(data: bytes, start: int, end: int) -> bytes:
    """Use ffmpeg to slice audio into a WAV chunk. Output is always WAV mono 16k.

    Uses temp files to avoid Windows pipe issues with large WAV streams.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(data)
        tmp_in_path = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-i", tmp_in_path,
             "-t", str(end - start), "-ar", "16000", "-ac", "1", tmp_out_path],
            capture_output=True, check=True,
        )
        return Path(tmp_out_path).read_bytes()
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


def chunk_audio(audio_bytes: bytes, *, max_seconds: int = 170, overlap_seconds: int = 10) -> list[AudioChunk]:
    """Split audio into chunks of <= max_seconds with overlap.

    Returns WAV chunks regardless of input format.
    """
    duration = _audio_duration_seconds(audio_bytes)
    if duration <= 180:  # within hard limit
        return [AudioChunk(audio_bytes=audio_bytes, time_start=0, time_end=int(duration),
                           chunk_index=0, n_total=1)]

    chunks: list[AudioChunk] = []
    start = 0
    while start < duration:
        end = min(start + max_seconds, int(duration))
        chunks.append(
            AudioChunk(
                audio_bytes=_slice_audio(audio_bytes, start, end),
                time_start=start,
                time_end=end,
                chunk_index=len(chunks),
                n_total=-1,
            )
        )
        if end >= duration:
            break
        start = end - overlap_seconds

    n = len(chunks)
    return [AudioChunk(audio_bytes=c.audio_bytes, time_start=c.time_start, time_end=c.time_end, chunk_index=c.chunk_index, n_total=n) for c in chunks]


@dataclass
class VideoChunk:
    video_bytes: bytes  # MP4 format
    time_start: int
    time_end: int
    chunk_index: int
    n_total: int


def _video_duration_seconds(data: bytes) -> float:
    """Probe video duration via ffprobe using a temp file (Windows pipe issue)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(tmp)],
            capture_output=True, check=True,
        )
        return float(p.stdout.decode().strip())
    finally:
        tmp.unlink(missing_ok=True)


def _slice_video(data: bytes, start: int, end: int) -> bytes:
    """Slice video to a temp file and return the bytes (Windows-safe)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
        f_in.write(data)
        tmp_in = Path(f_in.name)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_out:
        tmp_out = Path(f_out.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-i", str(tmp_in),
             "-t", str(end - start), "-c:v", "libx264", "-c:a", "aac",
             str(tmp_out)],
            capture_output=True, check=True,
        )
        return tmp_out.read_bytes()
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


def chunk_video(video_bytes: bytes, *, max_seconds: int = 70, overlap_seconds: int = 10) -> list[VideoChunk]:
    """Split video into chunks of <= max_seconds with overlap.

    Returns MP4 chunks regardless of input encoding.
    """
    duration = _video_duration_seconds(video_bytes)
    if duration <= 81:
        return [VideoChunk(video_bytes=video_bytes, time_start=0, time_end=int(duration),
                           chunk_index=0, n_total=1)]

    chunks: list[VideoChunk] = []
    start = 0
    while start < duration:
        end = min(start + max_seconds, int(duration))
        chunks.append(
            VideoChunk(
                video_bytes=_slice_video(video_bytes, start, end),
                time_start=start,
                time_end=end,
                chunk_index=len(chunks),
                n_total=-1,
            )
        )
        if end >= duration:
            break
        start = end - overlap_seconds

    n = len(chunks)
    return [VideoChunk(video_bytes=c.video_bytes, time_start=c.time_start, time_end=c.time_end, chunk_index=c.chunk_index, n_total=n) for c in chunks]
