# Gemini Embedding 2 + MongoDB Vector Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Educational demo: FastAPI app that ingests text, image, PDF, audio, and video files, embeds them with Vertex AI `gemini-embedding-2` at 1536 dims, stores in MongoDB Atlas Local with `$vectorSearch`, and returns multimodal semantic search results.

**Architecture:** Single-file FastAPI monolith (`app.py`) plus two helpers (`chunking.py`, `archives.py`). MongoDB Atlas Local in Docker for `$vectorSearch`. ADC for Vertex auth. Tests mock Vertex but use real Mongo.

**Tech Stack:** Python 3.12+, FastAPI, `google-genai` SDK, `pymongo`, `pypdf`, `ffmpeg-python`, `filetype`, `rarfile`, MongoDB Atlas Local 8.0.5, ffmpeg, unrar.

**Source spec:** [`docs/superpowers/specs/2026-05-11-gemini-embedding-2-mongo-design.md`](../specs/2026-05-11-gemini-embedding-2-mongo-design.md)

---

## File structure

| File | Responsibility | Created in task |
|---|---|---|
| `requirements.txt` | Pinned Python deps | T1 |
| `.env.example` | Environment variables template | T1 |
| `.gitignore` | Ignore `uploads/`, `.env`, `__pycache__/`, `.pytest_cache/`, `node_modules/`, `.venv/` | T1 |
| `docker-compose.yml` | `mongodb-atlas-local:8.0.5` service | T2 |
| `chunking.py` | `chunk_text` / `chunk_pdf` / `chunk_audio` / `chunk_video` | T4-T7 |
| `archives.py` | `extract_zip` / `extract_rar` with safety checks | T8-T9 |
| `vertex_client.py` | Wrapper around `google-genai`: `embed_doc` / `embed_query` / `count_tokens` | T10 |
| `mongo_setup.py` | Connection, index creation, healthcheck | T11 |
| `app.py` | FastAPI: routes + glue | T12-T18 |
| `templates/index.html` | Drag-and-drop UI + search box | T19-T20 |
| `static/styles.css` | Tailwind via CDN (no build) | T19 |
| `tests/conftest.py` | Pytest fixtures: mongo, mocked vertex, test client | T11, T12 |
| `tests/test_chunking.py` | Unit tests for `chunking.py` | T4-T7 |
| `tests/test_archives.py` | Unit tests for `archives.py` (incl. zip bomb, traversal) | T8-T9 |
| `tests/test_upload.py` | Integration tests for `/upload` | T13-T15 |
| `tests/test_search.py` | Integration tests for `/search` | T16-T17 |
| `tests/test_dedup.py` | Race / duplicate / re-upload | T15 |
| `tests/test_files_endpoint.py` | Security tests for `/files/{doc_id}` | T18 |
| `README.md` | Setup, smoke checklist, troubleshooting | T21 |
| `docs/00-scratchpad.md` | Crude notes (Cap. 2 ADK, Cap. 3 deploy, etc.) | T21 |

> Capítulos didácticos `docs/0X-*.md` se escriben **después** de tener el código corriendo (mejor entender antes de explicar). No están en este plan.

---

## Conventions

- **Python**: 3.12+ on Windows/Linux. Use `pathlib.Path`, never `os.path` strings.
- **Commits**: Conventional Commits in English, one commit per task. NEVER add Co-Authored-By or AI attribution.
- **Tests**: `pytest -v`. Mongo uses real container (started via `docker compose up -d mongo`). Vertex client is mocked via `pytest-mock` `MagicMock` returning fake 1536-dim vectors.
- **Sync-in-async**: pymongo is sync; FastAPI runs sync endpoints in threadpool. Routes are `def` (not `async def`) when they touch Mongo.
- **No emojis** anywhere.
- **Type hints**: required on all public functions.

---

## Phase 0 — Bootstrap

### Task 1: Project skeleton + dependencies

**Files:**
- Create: `requirements.txt`, `.env.example`, `.gitignore`, `pytest.ini`, `uploads/.gitkeep`, `templates/.gitkeep`, `static/.gitkeep`, `tests/__init__.py`

- [ ] **Step 1: Initialize git + venv**

```bash
cd C:/Users/smarq/IdeaProjects/gemini_embeddings_2
git init
python -m venv .venv
source .venv/Scripts/activate    # Git Bash on Windows
python --version                  # >= 3.12
```

- [ ] **Step 2: Write `requirements.txt`**

```
fastapi[standard]==0.115.*
jinja2==3.1.*
google-genai==1.*
pymongo==4.10.*
python-dotenv==1.0.*
pypdf==5.*
ffmpeg-python==0.2.*
filetype==1.2.*
rarfile==4.*
pytest==8.*
pytest-mock==3.*
httpx==0.28.*
```

- [ ] **Step 3: Write `.env.example`**

```
GCP_PROJECT=your-gcp-project-id
GCP_LOCATION=us-central1
MONGO_URI=mongodb://localhost:27017/?directConnection=true
MONGO_DB=embeddings_demo
MONGO_COLLECTION=documents
EMBEDDING_DIM=1536
EMBEDDING_MODEL=gemini-embedding-2
MAX_UPLOAD_MB=50
MAX_ARCHIVE_FILES=10
MAX_ARCHIVE_UNCOMPRESSED_MB=50
MAX_TOTAL_EMBED_SECONDS=1800
```

- [ ] **Step 4: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
uploads/*
!uploads/.gitkeep
.coverage
htmlcov/
*.egg-info/
node_modules/
```

- [ ] **Step 5: Write `pytest.ini`**

```
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra -v
asyncio_mode = auto
```

- [ ] **Step 6: Install deps**

```bash
pip install -r requirements.txt
pip list | grep -E "fastapi|pymongo|google-genai"
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example .gitignore pytest.ini uploads/.gitkeep templates/.gitkeep static/.gitkeep tests/__init__.py
git commit -m "chore: bootstrap project structure and pinned dependencies"
```

---

### Task 2: Docker Compose for MongoDB Atlas Local

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
services:
  mongo:
    image: mongodb/mongodb-atlas-local:8.0.5
    container_name: embeddings_demo_mongo
    ports:
      - "27017:27017"
    healthcheck:
      test: ["CMD", "mongosh", "--quiet", "--eval", "db.runCommand('ping').ok"]
      interval: 5s
      timeout: 3s
      retries: 10
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

- [ ] **Step 2: Start container and verify**

```bash
docker compose up -d mongo
docker compose ps
# Wait until STATUS shows "healthy"
docker compose logs mongo | tail -20
```

- [ ] **Step 3: Verify $vectorSearch availability via mongosh**

```bash
docker exec -it embeddings_demo_mongo mongosh --quiet --eval "db.adminCommand({listCommands:1}).commands.createSearchIndexes ? 'OK' : 'MISSING'"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: add docker-compose with mongodb-atlas-local 8.0.5"
```

---

## Phase 1 — Core libraries (TDD)

### Task 3: `vertex_client.py` — wrapper with mockable client

**Files:**
- Create: `vertex_client.py`, `tests/test_vertex_client.py`

- [ ] **Step 1: Write failing tests**

`tests/test_vertex_client.py`:

```python
from unittest.mock import MagicMock, patch
import pytest
from vertex_client import VertexClient, EmbeddingResult


@pytest.fixture
def mock_genai_client():
    """Fake google-genai Client. embed_content returns 1536-dim zero vector."""
    client = MagicMock()
    fake_response = MagicMock()
    fake_response.embeddings = [MagicMock(values=[0.0] * 1536)]
    client.models.embed_content.return_value = fake_response
    fake_count = MagicMock(total_tokens=42)
    client.models.count_tokens.return_value = fake_count
    return client


def test_embed_doc_passes_retrieval_document_task(mock_genai_client):
    vc = VertexClient(genai_client=mock_genai_client, model="gemini-embedding-2", dim=1536)
    result = vc.embed_doc(text="hello world")
    assert isinstance(result, EmbeddingResult)
    assert len(result.vector) == 1536
    call_args = mock_genai_client.models.embed_content.call_args
    cfg = call_args.kwargs["config"]
    assert cfg.task_type == "RETRIEVAL_DOCUMENT"
    assert cfg.output_dimensionality == 1536


def test_embed_query_passes_retrieval_query_task(mock_genai_client):
    vc = VertexClient(genai_client=mock_genai_client, model="gemini-embedding-2", dim=1536)
    vc.embed_query(text="search me")
    cfg = mock_genai_client.models.embed_content.call_args.kwargs["config"]
    assert cfg.task_type == "RETRIEVAL_QUERY"


def test_embed_doc_falls_back_when_task_type_rejected(mock_genai_client):
    # First call raises INVALID_ARGUMENT, second succeeds without task_type
    fake_response = MagicMock()
    fake_response.embeddings = [MagicMock(values=[0.1] * 1536)]
    mock_genai_client.models.embed_content.side_effect = [
        Exception("INVALID_ARGUMENT: task_type not supported"),
        fake_response,
    ]
    vc = VertexClient(genai_client=mock_genai_client, model="gemini-embedding-2", dim=1536)
    result = vc.embed_doc(text="hello")
    assert len(result.vector) == 1536
    assert mock_genai_client.models.embed_content.call_count == 2
    # Second call must not have task_type
    second_cfg = mock_genai_client.models.embed_content.call_args_list[1].kwargs["config"]
    assert second_cfg.task_type is None


def test_embed_doc_pdf_sets_document_ocr_flag(mock_genai_client):
    vc = VertexClient(genai_client=mock_genai_client, model="gemini-embedding-2", dim=1536)
    vc.embed_doc(file_bytes=b"%PDF-1.7", mime_type="application/pdf")
    cfg = mock_genai_client.models.embed_content.call_args.kwargs["config"]
    assert cfg.document_ocr is True


def test_count_tokens_returns_int(mock_genai_client):
    vc = VertexClient(genai_client=mock_genai_client, model="gemini-embedding-2", dim=1536)
    n = vc.count_tokens("some text")
    assert n == 42
```

- [ ] **Step 2: Run tests — expect import failure**

```bash
pytest tests/test_vertex_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'vertex_client'`

- [ ] **Step 3: Implement `vertex_client.py`**

```python
"""Wrapper around google-genai for embedding generation against Vertex AI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from google.genai import types


@dataclass
class EmbeddingResult:
    vector: list[float]
    task_type_used: Optional[str]
    flags: dict


class VertexClient:
    """Thin wrapper over google-genai Client.models.embed_content.

    Handles:
      - task_type best-effort with fallback if Vertex returns INVALID_ARGUMENT
      - per-modality flags (document_ocr for PDF, audio_track_extraction for video)
      - output_dimensionality (MRL truncation to `dim`)
    """

    def __init__(self, *, genai_client, model: str, dim: int):
        self._client = genai_client
        self._model = model
        self._dim = dim

    def embed_doc(
        self,
        *,
        text: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        mime_type: Optional[str] = None,
    ) -> EmbeddingResult:
        return self._embed(
            text=text,
            file_bytes=file_bytes,
            mime_type=mime_type,
            task_type="RETRIEVAL_DOCUMENT",
        )

    def embed_query(
        self,
        *,
        text: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        mime_type: Optional[str] = None,
    ) -> EmbeddingResult:
        return self._embed(
            text=text,
            file_bytes=file_bytes,
            mime_type=mime_type,
            task_type="RETRIEVAL_QUERY",
        )

    def count_tokens(self, text: str) -> int:
        resp = self._client.models.count_tokens(model=self._model, contents=[text])
        return resp.total_tokens

    def _build_contents(self, text, file_bytes, mime_type):
        if text is not None:
            return [text]
        if file_bytes is not None:
            return [types.Part.from_bytes(data=file_bytes, mime_type=mime_type)]
        raise ValueError("Provide either text or file_bytes")

    def _build_config(self, *, task_type, mime_type):
        flags = {}
        kwargs = {"output_dimensionality": self._dim}
        if task_type:
            kwargs["task_type"] = task_type
        if mime_type == "application/pdf":
            kwargs["document_ocr"] = True
            flags["document_ocr"] = True
        if mime_type and mime_type.startswith("video/"):
            kwargs["audio_track_extraction"] = True
            flags["audio_track_extraction"] = True
        return types.EmbedContentConfig(**kwargs), flags

    def _embed(self, *, text, file_bytes, mime_type, task_type) -> EmbeddingResult:
        contents = self._build_contents(text, file_bytes, mime_type)
        cfg, flags = self._build_config(task_type=task_type, mime_type=mime_type)
        try:
            resp = self._client.models.embed_content(
                model=self._model, contents=contents, config=cfg
            )
            return EmbeddingResult(
                vector=list(resp.embeddings[0].values),
                task_type_used=task_type,
                flags=flags,
            )
        except Exception as e:
            if "INVALID_ARGUMENT" in str(e) and task_type:
                # Fallback: retry without task_type (model may not honor it)
                cfg_no_task, _ = self._build_config(task_type=None, mime_type=mime_type)
                resp = self._client.models.embed_content(
                    model=self._model, contents=contents, config=cfg_no_task
                )
                return EmbeddingResult(
                    vector=list(resp.embeddings[0].values),
                    task_type_used=None,
                    flags=flags,
                )
            raise
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_vertex_client.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add vertex_client.py tests/test_vertex_client.py
git commit -m "feat(vertex): wrapper with task_type fallback and per-modality flags"
```

---

### Task 4: `chunking.py` — text chunking by tokens

**Files:**
- Create: `chunking.py`, `tests/test_chunking_text.py`

- [ ] **Step 1: Write failing tests**

`tests/test_chunking_text.py`:

```python
from unittest.mock import MagicMock
from chunking import chunk_text, TextChunk


def make_token_counter(tokens_per_chunk):
    """Returns a callable that simulates count_tokens.

    `tokens_per_chunk` maps text-prefix-length -> token count.
    For tests we use a simple word-based heuristic: ~1.5 tokens per word.
    """

    def _count(text: str) -> int:
        return max(1, int(len(text.split()) * 1.5))

    return _count


def test_short_text_returns_single_chunk():
    text = "Hello world. " * 100  # ~150 tokens
    counter = make_token_counter(None)
    chunks = chunk_text(text, count_tokens=counter, max_tokens=7000, overlap_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].n_total == 1


def test_long_text_splits_into_multiple_chunks():
    # 10000 words ~ 15000 tokens — should split into ~3 chunks
    text = " ".join(["palabra"] * 10_000)
    counter = make_token_counter(None)
    chunks = chunk_text(text, count_tokens=counter, max_tokens=7000, overlap_tokens=500)
    assert len(chunks) >= 2
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.n_total == len(chunks) for c in chunks)
    # chunk_index sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_have_overlap():
    text = " ".join([f"w{i}" for i in range(5000)])
    counter = make_token_counter(None)
    chunks = chunk_text(text, count_tokens=counter, max_tokens=2000, overlap_tokens=200)
    assert len(chunks) >= 2
    # The last words of chunk N should appear at the start of chunk N+1
    end_of_first = chunks[0].text.split()[-50:]
    start_of_second = chunks[1].text.split()[:50]
    overlap = set(end_of_first) & set(start_of_second)
    assert len(overlap) > 0, "expected token overlap between consecutive chunks"


def test_empty_text_returns_empty_list():
    counter = make_token_counter(None)
    assert chunk_text("", count_tokens=counter, max_tokens=7000, overlap_tokens=500) == []


def test_exactly_max_tokens_does_not_split():
    # Build a text whose token count equals exactly max_tokens
    text = " ".join(["word"] * 4666)  # 4666 * 1.5 = 6999 tokens
    counter = make_token_counter(None)
    chunks = chunk_text(text, count_tokens=counter, max_tokens=7000, overlap_tokens=500)
    assert len(chunks) == 1
```

- [ ] **Step 2: Run tests — expect import failure**

```bash
pytest tests/test_chunking_text.py -v
```

Expected: `ModuleNotFoundError: No module named 'chunking'`

- [ ] **Step 3: Implement `chunking.py` (text part)**

```python
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
        # Step forward leaving overlap
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_chunking_text.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add chunking.py tests/test_chunking_text.py
git commit -m "feat(chunking): text chunking by tokens with overlap"
```

---

### Task 5: `chunking.py` — PDF chunking by pages

**Files:**
- Modify: `chunking.py`
- Create: `tests/test_chunking_pdf.py`, `tests/fixtures/sample_15p.pdf` (15-page test PDF)

- [ ] **Step 1: Generate fixture PDF**

```bash
python -c "
from pypdf import PdfWriter
from pypdf.generic import NameObject, DictionaryObject, ArrayObject, FloatObject
w = PdfWriter()
for i in range(15):
    w.add_blank_page(width=200, height=200)
import os
os.makedirs('tests/fixtures', exist_ok=True)
with open('tests/fixtures/sample_15p.pdf', 'wb') as f:
    w.write(f)
print('OK')
"
```

- [ ] **Step 2: Write failing tests**

`tests/test_chunking_pdf.py`:

```python
from pathlib import Path
from chunking import chunk_pdf, PdfChunk

FIXTURE = Path("tests/fixtures/sample_15p.pdf")


def test_15_page_pdf_splits_into_4_chunks_with_overlap():
    """15 pages, max=4, overlap=1 → chunks of pages [1-4], [4-7], [7-10], [10-13], [13-15]."""
    chunks = chunk_pdf(FIXTURE.read_bytes(), max_pages=4, overlap_pages=1)
    assert all(isinstance(c, PdfChunk) for c in chunks)
    # Pages are 1-based inclusive
    page_ranges = [(c.page_start, c.page_end) for c in chunks]
    assert page_ranges == [(1, 4), (4, 7), (7, 10), (10, 13), (13, 15)]
    assert all(c.n_total == 5 for c in chunks)


def test_5_page_pdf_returns_single_chunk():
    """5 pages with max=4: a single chunk that fits within the model limit (≤6)."""
    from pypdf import PdfWriter
    import io
    w = PdfWriter()
    for _ in range(5):
        w.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    w.write(buf)
    chunks = chunk_pdf(buf.getvalue(), max_pages=4, overlap_pages=1)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 5
    assert chunks[0].chunk_index == 0


def test_each_chunk_bytes_is_valid_pdf():
    """Each chunk's bytes must parse as a valid PDF on its own (Vertex must read it)."""
    from pypdf import PdfReader
    import io
    chunks = chunk_pdf(FIXTURE.read_bytes(), max_pages=4, overlap_pages=1)
    for c in chunks:
        r = PdfReader(io.BytesIO(c.pdf_bytes))
        assert len(r.pages) == c.page_end - c.page_start + 1
```

- [ ] **Step 3: Run tests — expect failure**

```bash
pytest tests/test_chunking_pdf.py -v
```

Expected: `ImportError: cannot import name 'chunk_pdf'`

- [ ] **Step 4: Append to `chunking.py`**

```python
import io
from pypdf import PdfReader, PdfWriter


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
```

- [ ] **Step 5: Run tests — expect pass**

```bash
pytest tests/test_chunking_pdf.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add chunking.py tests/test_chunking_pdf.py tests/fixtures/sample_15p.pdf
git commit -m "feat(chunking): PDF page-range splitting with overlap"
```

---

### Task 6: `chunking.py` — audio chunking via ffmpeg

**Files:**
- Modify: `chunking.py`
- Create: `tests/test_chunking_audio.py`, `tests/fixtures/sample_300s.wav`

- [ ] **Step 1: Generate fixture audio (300s sine wave)**

```bash
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=300" -ar 16000 -ac 1 tests/fixtures/sample_300s.wav
```

- [ ] **Step 2: Write failing tests**

`tests/test_chunking_audio.py`:

```python
import subprocess
from pathlib import Path
from chunking import chunk_audio, AudioChunk


FIXTURE = Path("tests/fixtures/sample_300s.wav")


def _probe_duration_seconds(data: bytes) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", "-i", "pipe:0"],
        input=data, capture_output=True, check=True,
    )
    return float(p.stdout.decode().strip())


def test_300s_audio_splits_into_2_chunks():
    """300s audio with max=170s and overlap=10s → [0-170], [160-300]."""
    chunks = chunk_audio(FIXTURE.read_bytes(), max_seconds=170, overlap_seconds=10)
    assert len(chunks) == 2
    assert chunks[0].time_start == 0
    assert chunks[0].time_end == 170
    assert chunks[1].time_start == 160
    assert chunks[1].time_end == 300
    assert all(c.n_total == 2 for c in chunks)


def test_short_audio_returns_single_chunk():
    """120s audio (<= 180s hard limit) → single chunk passthrough."""
    short = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=120",
         "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    chunks = chunk_audio(short, max_seconds=170, overlap_seconds=10)
    assert len(chunks) == 1


def test_each_chunk_audio_is_valid():
    chunks = chunk_audio(FIXTURE.read_bytes(), max_seconds=170, overlap_seconds=10)
    for c in chunks:
        d = _probe_duration_seconds(c.audio_bytes)
        # ffmpeg reports +/- 0.5s of slack; assert within tolerance
        assert abs(d - (c.time_end - c.time_start)) < 1.0
```

- [ ] **Step 3: Run — expect failure**

```bash
pytest tests/test_chunking_audio.py -v
```

Expected: `ImportError: cannot import name 'chunk_audio'`

- [ ] **Step 4: Append to `chunking.py`**

```python
import subprocess


@dataclass
class AudioChunk:
    audio_bytes: bytes  # WAV format
    time_start: int     # seconds, 0-based
    time_end: int       # seconds, exclusive
    chunk_index: int
    n_total: int


def _audio_duration_seconds(data: bytes) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", "-i", "pipe:0"],
        input=data, capture_output=True, check=True,
    )
    return float(p.stdout.decode().strip())


def _slice_audio(data: bytes, start: int, end: int) -> bytes:
    """Use ffmpeg to slice audio into a WAV chunk. Output is always WAV mono 16k."""
    p = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", "pipe:0", "-t", str(end - start),
         "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
        input=data, capture_output=True, check=True,
    )
    return p.stdout


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
```

- [ ] **Step 5: Run — expect pass**

```bash
pytest tests/test_chunking_audio.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add chunking.py tests/test_chunking_audio.py tests/fixtures/sample_300s.wav
git commit -m "feat(chunking): audio chunking via ffmpeg with overlap"
```

---

### Task 7: `chunking.py` — video chunking via ffmpeg

**Files:**
- Modify: `chunking.py`
- Create: `tests/test_chunking_video.py`, `tests/fixtures/sample_200s.mp4`

- [ ] **Step 1: Generate fixture video (200s test pattern with audio)**

```bash
ffmpeg -y -f lavfi -i "testsrc=duration=200:size=320x240:rate=1" -f lavfi -i "sine=frequency=440:duration=200" -c:v libx264 -c:a aac -shortest tests/fixtures/sample_200s.mp4
```

- [ ] **Step 2: Write failing tests**

`tests/test_chunking_video.py`:

```python
import subprocess
from pathlib import Path
from chunking import chunk_video, VideoChunk


FIXTURE = Path("tests/fixtures/sample_200s.mp4")


def _probe_duration_seconds(data: bytes) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", "-i", "pipe:0"],
        input=data, capture_output=True, check=True,
    )
    return float(p.stdout.decode().strip())


def test_200s_video_splits_into_chunks_of_70s():
    """200s with max=70s and overlap=10s → [0-70], [60-130], [120-190], [180-200]."""
    chunks = chunk_video(FIXTURE.read_bytes(), max_seconds=70, overlap_seconds=10)
    assert len(chunks) == 4
    starts_ends = [(c.time_start, c.time_end) for c in chunks]
    assert starts_ends[0] == (0, 70)
    assert starts_ends[-1][1] == 200
    assert all(c.n_total == 4 for c in chunks)


def test_short_video_returns_single_chunk():
    short = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=30:size=320x240:rate=1",
         "-c:v", "libx264", "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    chunks = chunk_video(short, max_seconds=70, overlap_seconds=10)
    assert len(chunks) == 1


def test_each_chunk_video_is_valid_duration():
    chunks = chunk_video(FIXTURE.read_bytes(), max_seconds=70, overlap_seconds=10)
    for c in chunks:
        d = _probe_duration_seconds(c.video_bytes)
        assert abs(d - (c.time_end - c.time_start)) < 2.0
```

- [ ] **Step 3: Run — expect failure**

```bash
pytest tests/test_chunking_video.py -v
```

- [ ] **Step 4: Append to `chunking.py`**

```python
@dataclass
class VideoChunk:
    video_bytes: bytes  # MP4 format
    time_start: int
    time_end: int
    chunk_index: int
    n_total: int


def _video_duration_seconds(data: bytes) -> float:
    return _audio_duration_seconds(data)  # ffprobe handles both


def _slice_video(data: bytes, start: int, end: int) -> bytes:
    p = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", "pipe:0", "-t", str(end - start),
         "-c:v", "libx264", "-c:a", "aac",
         "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"],
        input=data, capture_output=True, check=True,
    )
    return p.stdout


def chunk_video(video_bytes: bytes, *, max_seconds: int = 70, overlap_seconds: int = 10) -> list[VideoChunk]:
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
```

- [ ] **Step 5: Run — expect pass**

```bash
pytest tests/test_chunking_video.py -v
```

- [ ] **Step 6: Commit**

```bash
git add chunking.py tests/test_chunking_video.py tests/fixtures/sample_200s.mp4
git commit -m "feat(chunking): video chunking via ffmpeg with overlap"
```

---

### Task 8: `archives.py` — ZIP extraction with safety guards

**Files:**
- Create: `archives.py`, `tests/test_archives_zip.py`

- [ ] **Step 1: Write failing tests**

`tests/test_archives_zip.py`:

```python
import io
import zipfile
import pytest
from archives import extract_archive, ArchiveEntry, ArchiveError


def _make_zip(entries: dict[str, bytes], compress=zipfile.ZIP_STORED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_extract_basic_zip_returns_entries():
    z = _make_zip({"a.txt": b"hello", "b.png": b"\x89PNG\r\n"})
    entries = extract_archive(z, mime_type="application/zip", max_files=10, max_uncompressed_mb=50)
    names = sorted(e.name for e in entries)
    assert names == ["a.txt", "b.png"]
    assert all(isinstance(e, ArchiveEntry) for e in entries)


def test_zip_bomb_rejected_pre_extraction():
    """Headers declare >50MB total — must reject WITHOUT extracting."""
    huge = b"x" * (10 * 1024 * 1024)  # 10MB
    z = _make_zip({f"f{i}.bin": huge for i in range(6)})  # declared 60MB total
    with pytest.raises(ArchiveError, match="exceeds.*uncompressed"):
        extract_archive(z, mime_type="application/zip", max_files=100, max_uncompressed_mb=50)


def test_too_many_files_rejected():
    z = _make_zip({f"f{i}.txt": b"x" for i in range(15)})
    with pytest.raises(ArchiveError, match="too many"):
        extract_archive(z, mime_type="application/zip", max_files=10, max_uncompressed_mb=50)


def test_path_traversal_entries_skipped():
    """Entries with `../` or absolute paths must be rejected."""
    z = _make_zip({"../etc/passwd": b"root:x:0", "good.txt": b"safe"})
    entries = extract_archive(z, mime_type="application/zip", max_files=10, max_uncompressed_mb=50)
    names = [e.name for e in entries]
    assert names == ["good.txt"]


def test_encrypted_zip_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zi = zipfile.ZipInfo("secret.txt")
        zi.flag_bits |= 0x1  # encrypted flag
        zf.writestr(zi, b"secret")
    with pytest.raises(ArchiveError, match="encrypted"):
        extract_archive(buf.getvalue(), mime_type="application/zip", max_files=10, max_uncompressed_mb=50)


def test_nested_archive_skipped_with_warning():
    inner = _make_zip({"inside.txt": b"x"})
    z = _make_zip({"nested.zip": inner, "ok.txt": b"y"})
    entries = extract_archive(z, mime_type="application/zip", max_files=10, max_uncompressed_mb=50)
    names = [e.name for e in entries]
    # nested.zip is skipped (depth=1 policy)
    assert "ok.txt" in names
    skipped = [e for e in entries if e.skipped]
    assert any(e.name == "nested.zip" for e in skipped)
```

- [ ] **Step 2: Run — expect failure**

```bash
pytest tests/test_archives_zip.py -v
```

Expected: `ModuleNotFoundError: No module named 'archives'`

- [ ] **Step 3: Implement `archives.py`**

```python
"""Safe extraction of ZIP/RAR archives with size, count, traversal, encryption guards."""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath


class ArchiveError(Exception):
    pass


@dataclass
class ArchiveEntry:
    name: str          # original (possibly nested) path inside archive
    data: bytes | None # None if skipped
    skipped: bool = False
    skip_reason: str | None = None


_NESTED_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"}


def _is_safe_path(name: str) -> bool:
    """Reject absolute paths, drive letters, and `..` traversal."""
    if not name or name.endswith("/"):
        return False
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute():
        return False
    if any(part == ".." for part in p.parts):
        return False
    if len(p.parts) > 0 and ":" in p.parts[0]:  # Windows drive letter
        return False
    return True


def extract_archive(
    data: bytes,
    *,
    mime_type: str,
    max_files: int,
    max_uncompressed_mb: int,
) -> list[ArchiveEntry]:
    if mime_type in ("application/zip", "application/x-zip-compressed"):
        return _extract_zip(data, max_files=max_files, max_uncompressed_mb=max_uncompressed_mb)
    if mime_type in ("application/x-rar", "application/vnd.rar", "application/x-rar-compressed"):
        return _extract_rar(data, max_files=max_files, max_uncompressed_mb=max_uncompressed_mb)
    raise ArchiveError(f"unsupported archive mime: {mime_type}")


def _extract_zip(data: bytes, *, max_files: int, max_uncompressed_mb: int) -> list[ArchiveEntry]:
    max_bytes = max_uncompressed_mb * 1024 * 1024
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ArchiveError(f"corrupt zip: {e}") from e

    infos = zf.infolist()

    # Guard 1: count
    real_files = [i for i in infos if not i.is_dir()]
    if len(real_files) > max_files:
        raise ArchiveError(f"too many files in archive: {len(real_files)} > {max_files}")

    # Guard 2: encrypted (any entry)
    for i in real_files:
        if i.flag_bits & 0x1:
            raise ArchiveError("encrypted archive entries not supported")

    # Guard 3: pre-extraction size sum (anti zip-bomb)
    total = sum(i.file_size for i in real_files)
    if total > max_bytes:
        raise ArchiveError(
            f"archive exceeds uncompressed cap: {total // (1024*1024)} MB > {max_uncompressed_mb} MB"
        )

    out: list[ArchiveEntry] = []
    for info in real_files:
        # Guard 4: path traversal
        if not _is_safe_path(info.filename):
            out.append(ArchiveEntry(name=info.filename, data=None, skipped=True,
                                    skip_reason="unsafe path"))
            continue
        # Guard 5: depth=1 — skip nested archives
        ext = os.path.splitext(info.filename)[1].lower()
        if ext in _NESTED_EXTS:
            out.append(ArchiveEntry(name=info.filename, data=None, skipped=True,
                                    skip_reason="nested archive (depth=1 policy)"))
            continue
        out.append(ArchiveEntry(name=info.filename, data=zf.read(info)))
    return out


def _extract_rar(data: bytes, *, max_files: int, max_uncompressed_mb: int) -> list[ArchiveEntry]:
    import rarfile
    max_bytes = max_uncompressed_mb * 1024 * 1024
    try:
        rf = rarfile.RarFile(io.BytesIO(data))
    except rarfile.Error as e:
        raise ArchiveError(f"corrupt rar: {e}") from e

    if rf.needs_password():
        raise ArchiveError("encrypted archive not supported")

    infos = [i for i in rf.infolist() if not i.isdir()]
    if len(infos) > max_files:
        raise ArchiveError(f"too many files: {len(infos)} > {max_files}")

    total = sum(i.file_size for i in infos)
    if total > max_bytes:
        raise ArchiveError(
            f"archive exceeds uncompressed cap: {total // (1024*1024)} MB > {max_uncompressed_mb} MB"
        )

    out: list[ArchiveEntry] = []
    for info in infos:
        if not _is_safe_path(info.filename):
            out.append(ArchiveEntry(name=info.filename, data=None, skipped=True,
                                    skip_reason="unsafe path"))
            continue
        ext = os.path.splitext(info.filename)[1].lower()
        if ext in _NESTED_EXTS:
            out.append(ArchiveEntry(name=info.filename, data=None, skipped=True,
                                    skip_reason="nested archive (depth=1 policy)"))
            continue
        out.append(ArchiveEntry(name=info.filename, data=rf.read(info)))
    return out
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_archives_zip.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add archives.py tests/test_archives_zip.py
git commit -m "feat(archives): zip extraction with anti-zip-bomb, traversal, encryption guards"
```

---

### Task 9: `archives.py` — RAR smoke test

**Files:**
- Create: `tests/test_archives_rar.py`, `tests/fixtures/sample.rar`

- [ ] **Step 1: Verify `unrar` binary available**

```bash
unrar 2>&1 | head -1 || echo "MISSING — install unrar via apt/brew/winget"
```

If missing on Windows: `winget install RARLab.WinRAR` (provides `unrar`).

- [ ] **Step 2: Generate fixture RAR (manually using rar tool)**

```bash
mkdir -p /tmp/rarfiles && echo "hello" > /tmp/rarfiles/a.txt && echo "world" > /tmp/rarfiles/b.txt
rar a tests/fixtures/sample.rar /tmp/rarfiles/a.txt /tmp/rarfiles/b.txt
```

- [ ] **Step 3: Write failing test**

`tests/test_archives_rar.py`:

```python
from pathlib import Path
import pytest
from archives import extract_archive, ArchiveError

FIXTURE = Path("tests/fixtures/sample.rar")


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture missing — see plan T9 step 2")
def test_extract_basic_rar():
    entries = extract_archive(
        FIXTURE.read_bytes(),
        mime_type="application/vnd.rar",
        max_files=10,
        max_uncompressed_mb=50,
    )
    names = sorted(e.name.split("/")[-1] for e in entries if not e.skipped)
    assert "a.txt" in names and "b.txt" in names
```

- [ ] **Step 4: Run — expect pass**

```bash
pytest tests/test_archives_rar.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_archives_rar.py tests/fixtures/sample.rar
git commit -m "test(archives): rar extraction smoke test"
```

---

## Phase 2 — Mongo wiring + healthcheck

### Task 10: `mongo_setup.py` — connection + indexes

**Files:**
- Create: `mongo_setup.py`, `tests/conftest.py`, `tests/test_mongo_setup.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
"""Shared fixtures for the test suite.

Mongo: real container at localhost:27017 (started via `docker compose up -d mongo`).
Vertex: mocked.
"""
import os
import uuid
from unittest.mock import MagicMock
import pytest
from pymongo import MongoClient


MONGO_URI = "mongodb://localhost:27017/?directConnection=true"


@pytest.fixture
def test_db_name():
    return f"test_emb_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def mongo_client():
    c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    c.admin.command("ping")  # fail fast if mongo down
    yield c
    c.close()


@pytest.fixture
def test_db(mongo_client, test_db_name):
    db = mongo_client[test_db_name]
    yield db
    mongo_client.drop_database(test_db_name)


@pytest.fixture
def fake_vector():
    """1536-dim fake embedding (random-ish but deterministic per call)."""
    import random
    rng = random.Random(42)
    return [rng.uniform(-1, 1) for _ in range(1536)]


@pytest.fixture
def mock_vertex_client(fake_vector):
    """Mocked VertexClient that returns the fake_vector for embed_doc/embed_query."""
    from vertex_client import EmbeddingResult
    vc = MagicMock()
    vc.embed_doc.return_value = EmbeddingResult(vector=fake_vector, task_type_used="RETRIEVAL_DOCUMENT", flags={})
    vc.embed_query.return_value = EmbeddingResult(vector=fake_vector, task_type_used="RETRIEVAL_QUERY", flags={})
    vc.count_tokens.return_value = 100
    return vc
```

- [ ] **Step 2: Write failing test**

`tests/test_mongo_setup.py`:

```python
from pymongo.errors import DuplicateKeyError
import pytest
from mongo_setup import init_indexes, healthcheck


def test_init_indexes_creates_dedup_unique(test_db):
    init_indexes(test_db, vector_dim=1536, collection_name="documents")
    idx = list(test_db["documents"].list_indexes())
    by_name = {i["name"]: i for i in idx}
    assert "dedup_idx" in by_name
    assert by_name["dedup_idx"]["unique"] is True


def test_unique_idx_blocks_concurrent_duplicate(test_db):
    init_indexes(test_db, vector_dim=1536, collection_name="documents")
    coll = test_db["documents"]
    base = {
        "content_hash": "sha256:abc",
        "embedding_model": "gemini-embedding-2",
        "embedding_dim": 1536,
        "chunk_index": 0,
    }
    coll.insert_one(base)
    with pytest.raises(DuplicateKeyError):
        coll.insert_one(dict(base))


def test_healthcheck_returns_ok_when_mongo_up(test_db, mongo_client):
    init_indexes(test_db, vector_dim=1536, collection_name="documents")
    h = healthcheck(test_db, collection_name="documents")
    assert h["mongo"] == "ok"
    assert h["dedup_index"] == "ready"
```

- [ ] **Step 3: Run — expect failure**

```bash
docker compose up -d mongo
sleep 5
pytest tests/test_mongo_setup.py -v
```

Expected: `ModuleNotFoundError: No module named 'mongo_setup'`

- [ ] **Step 4: Implement `mongo_setup.py`**

```python
"""MongoDB setup: connection helpers, index creation (vector + dedup), healthcheck."""
from __future__ import annotations

from pymongo.database import Database
from pymongo.operations import SearchIndexModel


def init_indexes(db: Database, *, vector_dim: int, collection_name: str = "documents") -> None:
    coll = db[collection_name]

    # Dedup B-tree unique index — race protection for concurrent uploads
    coll.create_index(
        [("content_hash", 1), ("embedding_model", 1), ("embedding_dim", 1), ("chunk_index", 1)],
        name="dedup_idx",
        unique=True,
    )

    # Atlas Search vector index
    existing = {ix.get("name") for ix in coll.list_search_indexes()}
    if "vector_index" not in existing:
        model = SearchIndexModel(
            definition={
                "fields": [
                    {"type": "vector", "path": "vector",
                     "numDimensions": vector_dim, "similarity": "cosine"},
                    {"type": "filter", "path": "modality"},
                    {"type": "filter", "path": "parent_doc_id"},
                    {"type": "filter", "path": "status"},
                ]
            },
            name="vector_index",
            type="vectorSearch",
        )
        coll.create_search_index(model=model)


def healthcheck(db: Database, *, collection_name: str = "documents") -> dict:
    out = {"mongo": "down", "dedup_index": "missing", "vector_index": "missing"}
    try:
        db.client.admin.command("ping")
        out["mongo"] = "ok"
    except Exception:
        return out
    coll = db[collection_name]
    if any(i["name"] == "dedup_idx" for i in coll.list_indexes()):
        out["dedup_index"] = "ready"
    try:
        if any(i.get("name") == "vector_index" for i in coll.list_search_indexes()):
            out["vector_index"] = "ready"
    except Exception:
        out["vector_index"] = "unsupported"
    return out
```

- [ ] **Step 5: Run — expect pass**

```bash
pytest tests/test_mongo_setup.py -v
```

Expected: 3 passed (vector index test may be slow on first run as Atlas Local builds the index).

- [ ] **Step 6: Commit**

```bash
git add mongo_setup.py tests/conftest.py tests/test_mongo_setup.py
git commit -m "feat(mongo): index init and healthcheck with unique dedup constraint"
```

---

## Phase 3 — FastAPI endpoints

### Task 11: `app.py` skeleton + `GET /health`

**Files:**
- Create: `app.py`, `tests/test_health.py`

- [ ] **Step 1: Write failing test**

`tests/test_health.py`:

```python
from fastapi.testclient import TestClient


def test_health_returns_status(monkeypatch, test_db_name):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017/?directConnection=true")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-2")
    from app import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mongo"] in ("ok", "down")
    assert "vector_index" in body
```

- [ ] **Step 2: Implement `app.py` skeleton**

```python
"""FastAPI demo: gemini-embedding-2 + MongoDB Atlas Local Vector Search.

Single-file monolith for didactic clarity. Helpers in chunking.py and archives.py.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient

from mongo_setup import healthcheck, init_indexes
from vertex_client import VertexClient

load_dotenv()

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?directConnection=true")
MONGO_DB = os.getenv("MONGO_DB", "embeddings_demo")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_ARCHIVE_FILES = int(os.getenv("MAX_ARCHIVE_FILES", "10"))
MAX_ARCHIVE_UNCOMPRESSED_MB = int(os.getenv("MAX_ARCHIVE_UNCOMPRESSED_MB", "50"))
MAX_TOTAL_EMBED_SECONDS = int(os.getenv("MAX_TOTAL_EMBED_SECONDS", "1800"))
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)


def _build_genai_client():
    """Initialize google-genai Client with Vertex AI backend."""
    from google import genai
    return genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    app.state.db = app.state.mongo[MONGO_DB]
    init_indexes(app.state.db, vector_dim=EMBEDDING_DIM, collection_name=MONGO_COLLECTION)
    if GCP_PROJECT:
        app.state.vertex = VertexClient(
            genai_client=_build_genai_client(),
            model=EMBEDDING_MODEL,
            dim=EMBEDDING_DIM,
        )
    else:
        app.state.vertex = None  # health endpoint still works without Vertex
    yield
    app.state.mongo.close()


app = FastAPI(lifespan=lifespan, title="gemini-embedding-2 demo")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/health")
def health():
    h = healthcheck(app.state.db, collection_name=MONGO_COLLECTION)
    h["vertex"] = "configured" if app.state.vertex else "missing GCP_PROJECT"
    return h
```

- [ ] **Step 3: Run — expect pass**

```bash
pytest tests/test_health.py -v
```

- [ ] **Step 4: Manual smoke**

```bash
uvicorn app:app --reload &
sleep 3
curl -s http://localhost:8000/health | python -m json.tool
kill %1
```

Expected: JSON with `mongo: "ok"`, `dedup_index: "ready"`, `vector_index: "ready"`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_health.py
git commit -m "feat(app): FastAPI skeleton with lifespan, settings, and /health endpoint"
```

---

### Task 12: Helpers — MIME sniffing, hashing, modality dispatch

**Files:**
- Modify: `app.py`
- Create: `tests/test_helpers.py`

- [ ] **Step 1: Write failing tests**

`tests/test_helpers.py`:

```python
import pytest
from app import sniff_mime, content_hash, modality_of, MODALITY_TEXT, MODALITY_IMAGE, MODALITY_PDF


def test_content_hash_is_sha256_prefixed():
    h = content_hash(b"hello")
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def test_sniff_mime_detects_png_from_bytes():
    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    assert sniff_mime(png_magic, fallback_name="x.png") == "image/png"


def test_sniff_mime_rejects_exe_renamed_to_pdf():
    """A binary that's actually MZ-prefixed (Windows exe) is NOT pdf."""
    exe = b"MZ\x90\x00" + b"\x00" * 100
    detected = sniff_mime(exe, fallback_name="evil.pdf")
    assert detected != "application/pdf"


def test_modality_of_maps_pdf():
    assert modality_of("application/pdf") == MODALITY_PDF


def test_modality_of_maps_image_variants():
    for m in ("image/png", "image/jpeg", "image/webp", "image/heic"):
        assert modality_of(m) == MODALITY_IMAGE


def test_modality_of_maps_audio_mp3_and_mpeg():
    assert modality_of("audio/mp3") == "audio"
    assert modality_of("audio/mpeg") == "audio"


def test_modality_of_unknown_returns_none():
    assert modality_of("application/x-msdownload") is None
```

- [ ] **Step 2: Append helpers to `app.py`**

```python
import hashlib
import filetype

MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"
MODALITY_PDF = "pdf"
MODALITY_AUDIO = "audio"
MODALITY_VIDEO = "video"

SUPPORTED_AUDIO = {"audio/mp3", "audio/mpeg", "audio/wav"}
SUPPORTED_VIDEO = {"video/mp4", "video/mpeg"}
SUPPORTED_IMAGE = {"image/png", "image/jpeg", "image/webp", "image/bmp",
                   "image/heic", "image/heif", "image/avif"}
SUPPORTED_PDF = {"application/pdf"}
SUPPORTED_TEXT_EXT = {".txt", ".md"}
SUPPORTED_ARCHIVE = {"application/zip", "application/x-zip-compressed",
                     "application/vnd.rar", "application/x-rar"}


def content_hash(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sniff_mime(data: bytes, *, fallback_name: str = "") -> str | None:
    """Return MIME from magic bytes, NOT from the upload's Content-Type header.

    Returns None for plain text (filetype can't sniff text). Caller must check extension.
    """
    kind = filetype.guess(data[:8192])
    if kind is not None:
        return kind.mime
    # Plain text fallback by extension
    ext = os.path.splitext(fallback_name.lower())[1]
    if ext in SUPPORTED_TEXT_EXT:
        return "text/plain"
    return None


def modality_of(mime: str | None) -> str | None:
    if mime is None:
        return None
    if mime in SUPPORTED_IMAGE:
        return MODALITY_IMAGE
    if mime in SUPPORTED_PDF:
        return MODALITY_PDF
    if mime in SUPPORTED_AUDIO:
        return MODALITY_AUDIO
    if mime in SUPPORTED_VIDEO:
        return MODALITY_VIDEO
    if mime.startswith("text/"):
        return MODALITY_TEXT
    return None
```

- [ ] **Step 3: Run — expect pass**

```bash
pytest tests/test_helpers.py -v
```

- [ ] **Step 4: Commit**

```bash
git add app.py tests/test_helpers.py
git commit -m "feat(app): MIME sniff, sha256 hash, modality dispatcher"
```

---

### Task 13: `POST /upload` for text + dedup

**Files:**
- Modify: `app.py`
- Create: `tests/test_upload_text.py`

- [ ] **Step 1: Write failing test**

`tests/test_upload_text.py`:

```python
from fastapi.testclient import TestClient


def _client(monkeypatch, test_db_name, mock_vertex):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-2")
    from app import app
    # Patch lifespan-injected vertex with our mock
    app.state.vertex = mock_vertex
    return TestClient(app)


def test_upload_txt_creates_one_chunk(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    r = c.post("/upload", files={"file": ("hello.txt", b"Hello world", "text/plain")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_chunks"] == 1
    assert body["modality"] == "text"
    assert "doc_id" in body


def test_upload_same_file_twice_returns_already_indexed(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    files = {"file": ("a.txt", b"same content", "text/plain")}
    r1 = c.post("/upload", files=files)
    assert r1.status_code == 200
    r2 = c.post("/upload", files={"file": ("a.txt", b"same content", "text/plain")})
    assert r2.status_code == 200
    assert r2.json()["status"] == "already_indexed"


def test_upload_unsupported_mime_returns_415(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    # MZ = Windows exe magic
    r = c.post("/upload", files={"file": ("evil.bin", b"MZ\x90\x00" + b"\x00" * 100, "application/octet-stream")})
    assert r.status_code == 415
```

- [ ] **Step 2: Append `/upload` to `app.py`**

```python
import uuid
from datetime import datetime, timezone
from fastapi import HTTPException, UploadFile, File
from pymongo.errors import DuplicateKeyError
from chunking import chunk_text


def _now():
    return datetime.now(timezone.utc)


def _ensure_within_size(data: bytes):
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail={"error": f"file > {MAX_UPLOAD_MB} MB"})


def _existing_doc(db, ch: str):
    return db[MONGO_COLLECTION].find_one(
        {"content_hash": ch, "embedding_model": EMBEDDING_MODEL,
         "embedding_dim": EMBEDDING_DIM, "chunk_index": 0},
        projection={"parent_doc_id": 1, "n_chunks_total": 1, "created_at": 1, "modality": 1},
    )


def _save_uploaded_file(data: bytes, original_name: str) -> tuple[str, Path]:
    """Save raw upload to disk under uploads/{uuid}/{sanitized_name}."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in original_name)[:120]
    doc_uuid = str(uuid.uuid4())
    folder = UPLOADS_DIR / doc_uuid
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / safe
    path.write_bytes(data)
    return doc_uuid, path


def _insert_chunks(db, docs: list[dict]):
    try:
        db[MONGO_COLLECTION].insert_many(docs, ordered=False)
    except DuplicateKeyError as e:
        # race: another request inserted the same doc — caller decides
        raise


def _ingest_text(db, *, vertex: VertexClient, raw_bytes: bytes, filename: str,
                 mime_type: str, content_h: str, storage_path: Path) -> dict:
    text = raw_bytes.decode("utf-8", errors="replace")
    chunks = chunk_text(text, count_tokens=vertex.count_tokens,
                        max_tokens=7000, overlap_tokens=500)
    if not chunks:
        raise HTTPException(status_code=422, detail={"error": "empty text"})
    parent_doc_id = str(uuid.uuid4())
    docs = []
    for ch in chunks:
        emb = vertex.embed_doc(text=ch.text)
        docs.append({
            "parent_doc_id": parent_doc_id,
            "chunk_index": ch.chunk_index,
            "n_chunks_total": ch.n_total,
            "modality": MODALITY_TEXT,
            "filename": filename,
            "mime_type": mime_type,
            "size_bytes": len(raw_bytes),
            "storage_path": str(storage_path),
            "preview_label": ch.text[:500],
            "content_hash": content_h,
            "chunk_meta": {"token_count": ch.token_count},
            "vector": emb.vector,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "embedding_task": emb.task_type_used,
            "embedding_flags": emb.flags,
            "status": "ok",
            "error": None,
            "retry_count": 0,
            "created_at": _now(),
        })
    _insert_chunks(db, docs)
    return {"doc_id": parent_doc_id, "n_chunks": len(chunks),
            "modality": MODALITY_TEXT, "status": "ok"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    raw = file.file.read()
    _ensure_within_size(raw)
    mime = sniff_mime(raw, fallback_name=file.filename or "")
    modality = modality_of(mime)
    if modality is None:
        raise HTTPException(status_code=415,
                            detail={"error": "unsupported mime",
                                    "detected_mime": mime,
                                    "supported": sorted(SUPPORTED_IMAGE | SUPPORTED_PDF |
                                                        SUPPORTED_AUDIO | SUPPORTED_VIDEO |
                                                        {"text/plain"})})

    ch = content_hash(raw)
    existing = _existing_doc(app.state.db, ch)
    if existing:
        return {"doc_id": existing["parent_doc_id"], "n_chunks": existing["n_chunks_total"],
                "modality": existing["modality"], "status": "already_indexed",
                "indexed_at": existing["created_at"].isoformat()}

    _, storage_path = _save_uploaded_file(raw, file.filename or "unnamed")

    if modality == MODALITY_TEXT:
        try:
            return _ingest_text(app.state.db, vertex=app.state.vertex,
                                raw_bytes=raw, filename=file.filename or "",
                                mime_type=mime, content_h=ch, storage_path=storage_path)
        except DuplicateKeyError:
            existing = _existing_doc(app.state.db, ch)
            return {"doc_id": existing["parent_doc_id"], "n_chunks": existing["n_chunks_total"],
                    "modality": existing["modality"], "status": "already_indexed"}

    raise HTTPException(status_code=501, detail={"error": f"modality {modality} not yet implemented"})
```

- [ ] **Step 3: Run — expect pass**

```bash
pytest tests/test_upload_text.py -v
```

- [ ] **Step 4: Commit**

```bash
git add app.py tests/test_upload_text.py
git commit -m "feat(upload): text ingestion with sha256 dedup and chunking"
```

---

### Task 14: `POST /upload` for binary modalities (image/PDF/audio/video)

**Files:**
- Modify: `app.py`
- Create: `tests/test_upload_binary.py`

- [ ] **Step 1: Write failing tests**

`tests/test_upload_binary.py`:

```python
from pathlib import Path
from fastapi.testclient import TestClient


def _client(monkeypatch, test_db_name, mock_vertex):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    from app import app
    app.state.vertex = mock_vertex
    return TestClient(app)


def test_upload_png_image_creates_single_chunk(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
    r = c.post("/upload", files={"file": ("a.png", png, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["modality"] == "image"
    assert body["n_chunks"] == 1


def test_upload_pdf_creates_chunks_per_page_block(monkeypatch, test_db_name, test_db, mock_vertex_client):
    fixture = Path("tests/fixtures/sample_15p.pdf").read_bytes()
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    r = c.post("/upload", files={"file": ("doc.pdf", fixture, "application/pdf")})
    assert r.status_code == 200
    assert r.json()["n_chunks"] == 5  # 15 pages / 4 with 1 overlap


def test_upload_audio_creates_chunks(monkeypatch, test_db_name, test_db, mock_vertex_client):
    fixture = Path("tests/fixtures/sample_300s.wav").read_bytes()
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    r = c.post("/upload", files={"file": ("a.wav", fixture, "audio/wav")})
    assert r.status_code == 200
    assert r.json()["modality"] == "audio"
    assert r.json()["n_chunks"] == 2


def test_upload_video_above_cost_cap_returns_413(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MAX_TOTAL_EMBED_SECONDS", "10")  # absurdly low
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    fixture = Path("tests/fixtures/sample_200s.mp4").read_bytes()
    r = c.post("/upload", files={"file": ("v.mp4", fixture, "video/mp4")})
    assert r.status_code == 413
```

- [ ] **Step 2: Extend `_ingest_*` functions in `app.py`**

```python
from chunking import chunk_pdf, chunk_audio, chunk_video, _audio_duration_seconds


def _base_chunk_doc(parent_id, ch_index, n_total, modality, filename, mime,
                    size, storage_path, preview, content_h, meta, emb):
    return {
        "parent_doc_id": parent_id, "chunk_index": ch_index, "n_chunks_total": n_total,
        "modality": modality, "filename": filename, "mime_type": mime, "size_bytes": size,
        "storage_path": str(storage_path), "preview_label": preview,
        "content_hash": content_h, "chunk_meta": meta,
        "vector": emb.vector, "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM, "embedding_task": emb.task_type_used,
        "embedding_flags": emb.flags, "status": "ok", "error": None,
        "retry_count": 0, "created_at": _now(),
    }


def _ingest_image(db, *, vertex, raw, filename, mime, content_h, storage_path):
    parent_id = str(uuid.uuid4())
    emb = vertex.embed_doc(file_bytes=raw, mime_type=mime)
    doc = _base_chunk_doc(parent_id, 0, 1, MODALITY_IMAGE, filename, mime,
                          len(raw), storage_path, f"image {mime}", content_h, {}, emb)
    _insert_chunks(db, [doc])
    return {"doc_id": parent_id, "n_chunks": 1, "modality": MODALITY_IMAGE, "status": "ok"}


def _ingest_pdf(db, *, vertex, raw, filename, mime, content_h, storage_path):
    chunks = chunk_pdf(raw, max_pages=4, overlap_pages=1)
    parent_id = str(uuid.uuid4())
    docs = []
    for ch in chunks:
        emb = vertex.embed_doc(file_bytes=ch.pdf_bytes, mime_type="application/pdf")
        docs.append(_base_chunk_doc(parent_id, ch.chunk_index, ch.n_total, MODALITY_PDF,
                                    filename, mime, len(raw), storage_path,
                                    f"PDF pages {ch.page_start}-{ch.page_end}",
                                    content_h, {"page_start": ch.page_start, "page_end": ch.page_end}, emb))
    _insert_chunks(db, docs)
    return {"doc_id": parent_id, "n_chunks": len(chunks), "modality": MODALITY_PDF, "status": "ok"}


def _ingest_av(db, *, vertex, raw, filename, mime, content_h, storage_path, modality):
    duration = _audio_duration_seconds(raw)
    if duration > MAX_TOTAL_EMBED_SECONDS:
        raise HTTPException(status_code=413,
                            detail={"error": "embedding budget exceeded",
                                    "limit_seconds": MAX_TOTAL_EMBED_SECONDS,
                                    "requested_seconds": int(duration)})
    if modality == MODALITY_AUDIO:
        chunks = chunk_audio(raw, max_seconds=170, overlap_seconds=10)
        chunk_bytes_attr = "audio_bytes"
        media_mime = "audio/wav"
    else:
        chunks = chunk_video(raw, max_seconds=70, overlap_seconds=10)
        chunk_bytes_attr = "video_bytes"
        media_mime = "video/mp4"
    parent_id = str(uuid.uuid4())
    docs = []
    for ch in chunks:
        emb = vertex.embed_doc(file_bytes=getattr(ch, chunk_bytes_attr), mime_type=media_mime)
        docs.append(_base_chunk_doc(parent_id, ch.chunk_index, ch.n_total, modality,
                                    filename, mime, len(raw), storage_path,
                                    f"{modality} {ch.time_start}-{ch.time_end}s",
                                    content_h, {"time_start": ch.time_start, "time_end": ch.time_end}, emb))
    _insert_chunks(db, docs)
    return {"doc_id": parent_id, "n_chunks": len(chunks), "modality": modality, "status": "ok"}
```

Then replace the `if modality == MODALITY_TEXT` block in the `/upload` handler with:

```python
    try:
        kwargs = dict(raw=raw, filename=file.filename or "", mime=mime,
                      content_h=ch, storage_path=storage_path, vertex=app.state.vertex)
        if modality == MODALITY_TEXT:
            return _ingest_text(app.state.db, raw_bytes=raw, **{k: v for k, v in kwargs.items() if k != "raw"})
        if modality == MODALITY_IMAGE:
            return _ingest_image(app.state.db, **kwargs)
        if modality == MODALITY_PDF:
            return _ingest_pdf(app.state.db, **kwargs)
        if modality in (MODALITY_AUDIO, MODALITY_VIDEO):
            return _ingest_av(app.state.db, modality=modality, **kwargs)
    except DuplicateKeyError:
        existing = _existing_doc(app.state.db, ch)
        return {"doc_id": existing["parent_doc_id"], "n_chunks": existing["n_chunks_total"],
                "modality": existing["modality"], "status": "already_indexed"}
    raise HTTPException(status_code=415, detail={"error": "unhandled modality"})
```

> Note: `_ingest_text` keeps its `raw_bytes=` parameter; the rename is intentional since text decodes the bytes.

- [ ] **Step 3: Run — expect pass**

```bash
pytest tests/test_upload_binary.py -v
```

- [ ] **Step 4: Commit**

```bash
git add app.py tests/test_upload_binary.py
git commit -m "feat(upload): image, PDF, audio, video ingestion with cost guard"
```

---

### Task 15: `POST /upload` archive support (ZIP/RAR)

**Files:**
- Modify: `app.py`
- Create: `tests/test_upload_archive.py`

- [ ] **Step 1: Write failing test**

`tests/test_upload_archive.py`:

```python
import io, zipfile
from pathlib import Path
from fastapi.testclient import TestClient


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for k, v in entries.items():
            zf.writestr(k, v)
    return buf.getvalue()


def _client(monkeypatch, test_db_name, mock_vertex):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    from app import app
    app.state.vertex = mock_vertex
    return TestClient(app)


def test_upload_zip_with_mix_returns_per_file_summary(monkeypatch, test_db_name, test_db, mock_vertex_client):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    z = _zip({"a.txt": b"hello", "b.png": png, "evil.exe": b"MZ\x90\x00" + b"\x00" * 100})
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    r = c.post("/upload", files={"file": ("mix.zip", z, "application/zip")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "extracted" in body
    statuses = {e["filename"]: e["status"] for e in body["extracted"]}
    assert statuses["a.txt"] == "ok"
    assert statuses["b.png"] == "ok"
    assert statuses["evil.exe"] == "skipped"
    assert body["summary"]["ok"] == 2
    assert body["summary"]["skipped"] == 1
```

- [ ] **Step 2: Append to `app.py`**

```python
from archives import extract_archive, ArchiveError, ArchiveEntry


def _ingest_extracted(db, *, vertex, entry: ArchiveEntry, archive_filename: str):
    """Process a single file from inside an archive. Returns a per-file result dict."""
    if entry.skipped:
        return {"filename": entry.name, "status": "skipped", "reason": entry.skip_reason}
    raw = entry.data
    mime = sniff_mime(raw, fallback_name=entry.name)
    modality = modality_of(mime)
    if modality is None:
        return {"filename": entry.name, "status": "skipped", "reason": f"unsupported mime: {mime}"}
    ch = content_hash(raw)
    existing = _existing_doc(db, ch)
    if existing:
        return {"filename": entry.name, "doc_id": existing["parent_doc_id"], "status": "already_indexed"}
    _, storage_path = _save_uploaded_file(raw, entry.name)
    try:
        kwargs = dict(raw=raw, filename=entry.name, mime=mime, content_h=ch,
                      storage_path=storage_path, vertex=vertex)
        if modality == MODALITY_TEXT:
            res = _ingest_text(db, raw_bytes=raw, vertex=vertex, filename=entry.name,
                               mime_type=mime, content_h=ch, storage_path=storage_path)
        elif modality == MODALITY_IMAGE:
            res = _ingest_image(db, **kwargs)
        elif modality == MODALITY_PDF:
            res = _ingest_pdf(db, **kwargs)
        else:
            res = _ingest_av(db, modality=modality, **kwargs)
    except DuplicateKeyError:
        existing = _existing_doc(db, ch)
        return {"filename": entry.name, "doc_id": existing["parent_doc_id"], "status": "already_indexed"}
    # Mark source_archive on inserted docs
    db[MONGO_COLLECTION].update_many(
        {"parent_doc_id": res["doc_id"]},
        {"$set": {"source_archive": {"filename": archive_filename, "extracted_at": entry.name}}},
    )
    return {"filename": entry.name, "doc_id": res["doc_id"], "n_chunks": res["n_chunks"], "status": "ok"}
```

In the `/upload` handler, BEFORE the existing modality dispatch, add archive handling:

```python
    if mime in SUPPORTED_ARCHIVE:
        try:
            entries = extract_archive(raw, mime_type=mime,
                                      max_files=MAX_ARCHIVE_FILES,
                                      max_uncompressed_mb=MAX_ARCHIVE_UNCOMPRESSED_MB)
        except ArchiveError as e:
            raise HTTPException(status_code=413, detail={"error": str(e)})
        results = [_ingest_extracted(app.state.db, vertex=app.state.vertex,
                                     entry=e, archive_filename=file.filename or "")
                   for e in entries]
        summary = {"total": len(results),
                   "ok": sum(1 for r in results if r["status"] == "ok"),
                   "already_indexed": sum(1 for r in results if r["status"] == "already_indexed"),
                   "skipped": sum(1 for r in results if r["status"] == "skipped"),
                   "failed": sum(1 for r in results if r["status"] == "failed")}
        return {"archive": file.filename, "extracted": results, "summary": summary}
```

Add `SUPPORTED_ARCHIVE` to the modality_of bypass — archives are detected before MIME→modality.

- [ ] **Step 3: Run — expect pass**

```bash
pytest tests/test_upload_archive.py -v
```

- [ ] **Step 4: Commit**

```bash
git add app.py tests/test_upload_archive.py
git commit -m "feat(upload): zip/rar extraction with per-file ingestion summary"
```

---

### Task 16: `POST /search` text query

**Files:**
- Modify: `app.py`
- Create: `tests/test_search_text.py`

- [ ] **Step 1: Write failing test**

`tests/test_search_text.py`:

```python
from fastapi.testclient import TestClient


def _client(monkeypatch, test_db_name, mock_vertex):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    from app import app
    app.state.vertex = mock_vertex
    return TestClient(app)


def test_search_text_query_returns_results_grouped_by_parent(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    # Seed 3 docs
    for content in [b"alpha text", b"beta text", b"gamma text"]:
        c.post("/upload", files={"file": (f"{content[:5].decode()}.txt", content, "text/plain")})
    r = c.post("/search", json={"query": "anything"})
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    # Vector search index can be eventually-consistent — at minimum response shape is valid
    assert isinstance(body["results"], list)


def test_search_with_empty_payload_returns_400(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    r = c.post("/search", json={})
    assert r.status_code == 400


def test_search_filters_by_modality(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    r = c.post("/search", json={"query": "x", "modality": ["pdf", "text"]})
    assert r.status_code == 200
```

- [ ] **Step 2: Append `/search` to `app.py`**

```python
from typing import Optional
from pydantic import BaseModel
from fastapi import Form, Body


class SearchPayload(BaseModel):
    query: Optional[str] = None
    modality: Optional[list[str]] = None
    limit: int = 10


@app.post("/search")
def search(payload: SearchPayload = Body(...)):
    if not payload.query:
        raise HTTPException(status_code=400, detail={"error": "provide query or file"})
    emb = app.state.vertex.embed_query(text=payload.query)
    return _vector_search(emb.vector, modality_filter=payload.modality, limit=payload.limit)


@app.post("/search/file")
def search_file(file: UploadFile = File(...)):
    raw = file.file.read()
    _ensure_within_size(raw)
    mime = sniff_mime(raw, fallback_name=file.filename or "")
    if modality_of(mime) is None:
        raise HTTPException(status_code=415, detail={"error": "unsupported mime"})
    emb = app.state.vertex.embed_query(file_bytes=raw, mime_type=mime)
    return _vector_search(emb.vector, modality_filter=None, limit=10)


def _vector_search(query_vector: list[float], *, modality_filter, limit: int) -> dict:
    vfilter: dict = {"status": "ok"}
    if modality_filter:
        vfilter["modality"] = {"$in": modality_filter}
    pipeline = [
        {"$vectorSearch": {
            "index": "vector_index", "path": "vector", "queryVector": query_vector,
            "numCandidates": limit * 20, "limit": limit * 2, "filter": vfilter,
        }},
        {"$project": {"vector": 0,
                      "score": {"$meta": "vectorSearchScore"},
                      "parent_doc_id": 1, "chunk_index": 1, "modality": 1,
                      "filename": 1, "preview_label": 1, "chunk_meta": 1}},
        {"$sort": {"parent_doc_id": 1, "score": -1}},
        {"$group": {"_id": "$parent_doc_id",
                    "best_score": {"$first": "$score"},
                    "best_chunk": {"$first": "$$ROOT"},
                    "matched_chunks": {"$sum": 1}}},
        {"$sort": {"best_score": -1}},
        {"$limit": limit},
    ]
    out = list(app.state.db[MONGO_COLLECTION].aggregate(pipeline))
    return {"results": [{
        "doc_id": r["_id"],
        "score": r["best_score"],
        "matched_chunks": r["matched_chunks"],
        "best_chunk": {k: v for k, v in r["best_chunk"].items() if k != "_id"},
    } for r in out]}
```

- [ ] **Step 3: Run — expect pass**

```bash
pytest tests/test_search_text.py -v
```

> Note: Atlas Local vector index is eventually consistent; the seeded docs may not appear immediately. Tests assert response *shape*, not specific matches. Manual smoke validates relevance.

- [ ] **Step 4: Commit**

```bash
git add app.py tests/test_search_text.py
git commit -m "feat(search): text and file query endpoints with vector search aggregation"
```

---

### Task 17: `GET /files/{doc_id}` secure file serving

**Files:**
- Modify: `app.py`
- Create: `tests/test_files_endpoint.py`

- [ ] **Step 1: Write failing test**

`tests/test_files_endpoint.py`:

```python
from fastapi.testclient import TestClient


def _client(monkeypatch, test_db_name, mock_vertex):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    from app import app
    app.state.vertex = mock_vertex
    return TestClient(app)


def test_files_endpoint_returns_attachment(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    r = c.post("/upload", files={"file": ("hello.txt", b"hello", "text/plain")})
    doc_id = r.json()["doc_id"]
    r2 = c.get(f"/files/{doc_id}")
    assert r2.status_code == 200
    assert r2.headers["content-disposition"].startswith("attachment;")


def test_files_endpoint_404_for_unknown(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    r = c.get("/files/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_files_endpoint_rejects_path_traversal(monkeypatch, test_db_name, test_db, mock_vertex_client):
    c = _client(monkeypatch, test_db_name, mock_vertex_client)
    r = c.get("/files/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code in (400, 404, 422)
```

- [ ] **Step 2: Append to `app.py`**

```python
from fastapi.responses import FileResponse


@app.get("/files/{doc_id}")
def serve_file(doc_id: str):
    # Validate doc_id is a UUID-ish string (no path chars allowed)
    if not all(c.isalnum() or c == "-" for c in doc_id) or len(doc_id) > 64:
        raise HTTPException(status_code=400, detail={"error": "invalid doc_id"})
    doc = app.state.db[MONGO_COLLECTION].find_one({"parent_doc_id": doc_id},
                                                  projection={"storage_path": 1, "filename": 1, "mime_type": 1})
    if not doc:
        raise HTTPException(status_code=404, detail={"error": "doc not found"})
    p = Path(doc["storage_path"]).resolve()
    uploads_real = UPLOADS_DIR.resolve()
    # Defense in depth: ensure resolved path is under uploads/
    if not str(p).startswith(str(uploads_real)):
        raise HTTPException(status_code=400, detail={"error": "invalid storage_path"})
    if not p.exists():
        raise HTTPException(status_code=404, detail={"error": "file missing on disk"})
    return FileResponse(path=str(p), media_type=doc["mime_type"],
                        filename=doc["filename"],
                        headers={"Content-Disposition": f'attachment; filename="{doc["filename"]}"'})
```

- [ ] **Step 3: Run — expect pass**

```bash
pytest tests/test_files_endpoint.py -v
```

- [ ] **Step 4: Commit**

```bash
git add app.py tests/test_files_endpoint.py
git commit -m "feat(files): secure file download with path traversal guards"
```

---

## Phase 4 — UI (manual smoke testing)

### Task 18: `templates/index.html` — drag-and-drop + search

**Files:**
- Create: `templates/index.html`, `static/styles.css` (empty placeholder; we use Tailwind CDN)
- Modify: `app.py` (add `GET /` route)

- [ ] **Step 1: Add root route to `app.py`**

```python
from fastapi import Request


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
```

- [ ] **Step 2: Write `templates/index.html`**

```html
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>gemini-embedding-2 demo</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-100 text-slate-900 font-sans">
  <main class="max-w-3xl mx-auto p-8 space-y-8">
    <header>
      <h1 class="text-3xl font-bold">gemini-embedding-2 + MongoDB Vector</h1>
      <p class="text-sm text-slate-600">Sube cualquier archivo (texto, imagen, PDF, audio, video, ZIP, RAR) y bus&caacute;calo sem&aacute;nticamente.</p>
    </header>

    <section>
      <h2 class="text-xl font-semibold mb-2">Subir archivo</h2>
      <div id="dropzone" class="border-2 border-dashed border-slate-400 rounded p-8 text-center cursor-pointer hover:border-slate-600">
        Arrastra un archivo aqu&iacute; o haz clic para seleccionar
        <input id="fileInput" type="file" class="hidden" />
      </div>
      <div id="uploadStatus" class="mt-3 text-sm"></div>
    </section>

    <section>
      <h2 class="text-xl font-semibold mb-2">Buscar</h2>
      <form id="searchForm" class="flex gap-2">
        <input id="queryInput" type="text" class="flex-1 border rounded px-3 py-2"
               placeholder='ej: "perro corriendo en la playa"' />
        <button type="submit" class="bg-slate-900 text-white px-4 py-2 rounded">Buscar</button>
      </form>
      <div id="searchResults" class="mt-4 space-y-3"></div>
    </section>
  </main>

  <script>
    const dz = document.getElementById('dropzone');
    const input = document.getElementById('fileInput');
    const status = document.getElementById('uploadStatus');

    dz.addEventListener('click', () => input.click());
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('bg-slate-200'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('bg-slate-200'));
    dz.addEventListener('drop', e => {
      e.preventDefault();
      dz.classList.remove('bg-slate-200');
      if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => {
      if (input.files.length) handleUpload(input.files[0]);
    });

    async function handleUpload(file) {
      status.textContent = `Subiendo ${file.name}...`;
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch('/upload', { method: 'POST', body: fd });
      const body = await r.json();
      if (!r.ok) {
        status.textContent = `Error ${r.status}: ${JSON.stringify(body.detail || body)}`;
        return;
      }
      if (body.archive) {
        status.innerHTML = `Archivo <b>${body.archive}</b>: ${body.summary.ok} ok, ${body.summary.already_indexed} ya indexados, ${body.summary.skipped} omitidos.`;
      } else {
        status.innerHTML = `<b>${body.status}</b> &mdash; ${body.modality}, ${body.n_chunks} chunk(s). doc_id: <code>${body.doc_id}</code>`;
      }
    }

    document.getElementById('searchForm').addEventListener('submit', async e => {
      e.preventDefault();
      const q = document.getElementById('queryInput').value;
      const r = await fetch('/search', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, limit: 10 })
      });
      const body = await r.json();
      const out = document.getElementById('searchResults');
      out.innerHTML = '';
      if (!body.results || body.results.length === 0) {
        out.innerHTML = '<p class="text-slate-500">Sin resultados.</p>';
        return;
      }
      for (const r of body.results) {
        const c = r.best_chunk;
        const pct = Math.round(r.score * 100);
        const div = document.createElement('div');
        div.className = 'border rounded p-3 bg-white';
        div.innerHTML = `
          <div class="flex justify-between text-sm text-slate-600">
            <span><b>${c.modality}</b> &middot; ${c.filename}</span>
            <span>${pct}% &middot; ${r.matched_chunks} chunk(s)</span>
          </div>
          <p class="mt-1 text-sm">${(c.preview_label || '').slice(0, 250)}</p>
          <a class="text-blue-600 text-xs" href="/files/${r.doc_id}">descargar</a>
        `;
        out.appendChild(div);
      }
    });
  </script>
</body>
</html>
```

- [ ] **Step 3: Empty `static/styles.css`** (place-holder so `StaticFiles` mount works)

```bash
touch static/styles.css
```

- [ ] **Step 4: Manual smoke test**

```bash
docker compose up -d mongo
uvicorn app:app --reload &
sleep 3
xdg-open http://localhost:8000 || start http://localhost:8000
```

Manual checks:
1. Drag a `.txt` file → status shows `ok` with doc_id
2. Drag the same file again → status shows `already_indexed`
3. Type a search query → results render
4. Click "descargar" on a result → file downloads with correct filename

- [ ] **Step 5: Commit**

```bash
git add app.py templates/index.html static/styles.css
git commit -m "feat(ui): drag-and-drop upload and semantic search HTML"
```

---

## Phase 5 — Documentation

### Task 19: README + scratchpad + smoke checklist

**Files:**
- Create: `README.md`, `docs/00-scratchpad.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# gemini-embedding-2 + MongoDB Vector Search demo

Educational demo: upload any file (text, image, PDF, audio, video, ZIP, RAR) and search them semantically using `gemini-embedding-2` embeddings stored in MongoDB Atlas Local.

## Prerequisites

- Python 3.12+
- Docker (for MongoDB Atlas Local)
- ffmpeg (`apt install ffmpeg` / `brew install ffmpeg` / `winget install ffmpeg`)
- unrar (`apt install unrar` / `brew install rar` / `winget install RARLab.WinRAR`)
- Google Cloud SDK with `gcloud auth application-default login` completed
- A GCP project with Vertex AI API enabled

## Setup

```bash
git clone <this-repo> && cd gemini_embeddings_2
python -m venv .venv && source .venv/Scripts/activate    # or .venv/bin/activate on Linux/Mac
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GCP_PROJECT to your project ID
docker compose up -d mongo
gcloud auth application-default login
uvicorn app:app --reload
```

Open http://localhost:8000

## Smoke checklist (pre-recording)

1. `GET /health` returns `{mongo: "ok", dedup_index: "ready", vector_index: "ready", vertex: "configured"}`
2. Upload 3 short `.txt` (cats, soccer, JS), search `"animales domésticos"` → top result is the cats file
3. Upload images (beach, dog, pizza), search `"comida italiana"` → top result is pizza
4. Upload a 4-page scanned PDF, search text only present in the scanned image → matches (OCR works)
5. Upload a 15-page PDF → response shows `n_chunks: 5`
6. Upload the same `.txt` twice → second response is `status: "already_indexed"`
7. Upload a ZIP with `.txt + .png + .pdf + .exe` → 3 ok, 1 skipped
8. Upload a 30-second `.mp3` → search text from the audio → matches
9. Upload an `.exe` directly → 415 with explicit error
10. Upload a corrupt PDF → 422

## Running tests

```bash
docker compose up -d mongo
pytest -v
```

Tests mock Vertex AI but use real MongoDB. They create disposable databases per test and clean up.

## Troubleshooting

- **`vector_index: "missing"`** — Atlas Local builds the index on first start; wait ~30s and retry `/health`.
- **`Vertex AI: 403`** — `gcloud auth application-default login` not completed, or Vertex AI API not enabled in the project.
- **`ffmpeg: not found`** — install ffmpeg system-wide; not pip-installable.
- **`unrar: not found`** — install via system package manager. Without it, `.rar` uploads fail with 500.
- **`vector_search: command not recognized`** — you ran the standard `mongo` image instead of `mongodb/mongodb-atlas-local`. Check `docker-compose.yml`.

## Architecture references

- Design spec: `docs/superpowers/specs/2026-05-11-gemini-embedding-2-mongo-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-12-gemini-embedding-2-mongo-implementation.md`
````

- [ ] **Step 2: Write `docs/00-scratchpad.md`**

```markdown
# Scratchpad — ideas, dudas, anotaciones crudas

## Ideas para capítulos posteriores (Cap. 2 / Cap. 3)

- **Cap. 2:** agente conversacional con Google ADK encima del vector search.
- **Cap. 3:** deploy a Cloud Run + job queue async (Cloud Tasks o BackgroundTasks + `/jobs/{id}` polling) para uploads grandes sin cap.

## Mejoras técnicas v2 (no MVP)

- Embeddings paralelos con `asyncio.gather` (cuidado con quota Vertex).
- Batch de hasta 6 imágenes en una sola llamada (optimización costo).
- Near-duplicate detection con MinHash/SimHash.
- Endpoint `DELETE /files/{doc_id}`.
- Cambiar `pymongo` → `motor` para async puro.

## Pre-grabación del video

- Pre-warm de quota Vertex en `us-central1`: correr 5-10 embeddings de prueba antes de grabar.
- Considerar pedir cuota-bump a GCP si la cuenta es nueva.
- Tener fixtures de prueba listos: 3 .txt, 3 imágenes, 1 PDF escaneado, 1 PDF largo, 1 mp3 corto.

## Comparativa para video (datos a recolectar)

- `text-embedding-3-large` (OpenAI) en mismo set de prueba.
- `voyage-multimodal-3` (Voyage) si hay tiempo.
- `cohere-embed-v3` (Cohere) si hay tiempo.

## Dudas pendientes

- ¿`task_type` finalmente se honra o se ignora silenciosamente en `gemini-embedding-2`? Validar empíricamente con embeddings A/B (mismo input, una con task una sin → comparar coseno).
- ¿El `vectorSearchScore` normalizado [0,1] depende de la dimensión o es estable cross-dim?
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/00-scratchpad.md
git commit -m "docs: README with smoke checklist and scratchpad"
```

---

## Self-Review (engineer-side checklist)

After all tasks: verify with one round of full pytest + manual smoke.

- [ ] Run all tests: `pytest -v` — expect green
- [ ] Run manual smoke checklist (README section "Smoke checklist") top-to-bottom
- [ ] Verify `/health` returns all green
- [ ] Verify the spec's Section 7 cases 1-8 each work end-to-end
- [ ] Verify no `TODO`/`TBD`/`FIXME` left in code (`grep -rn "TODO\|TBD\|FIXME" *.py templates/`)

---

## Plan complete

**Plan saved to:** `docs/superpowers/plans/2026-05-12-gemini-embedding-2-mongo-implementation.md`

### Spec coverage map

| Spec section | Implemented in tasks |
|---|---|
| §2 Architecture | T1, T11 |
| §3 Components / deps / .env | T1, T2 |
| §4 Upload flow + chunking | T4-T7, T13, T14 |
| §4 Search flow | T16 |
| §4 Modality limits | T6, T7 enforcement; T12 sniff |
| §5 Mongo schema + index | T10, T13-T15 (writes) |
| §5 Aggregation pipeline | T16 |
| §6 Errors (415/413/422/502) | T13 (415), T14 (413 cost), T15 (archive errors) |
| §6 Idempotencia + race | T10 (unique idx), T13/T14/T15 (DuplicateKeyError catch) |
| §6 Archive safety | T8, T9 |
| §6 Security uploads | T17 |
| §7 Tests automatizados | conftest in T10, per-feature tests T3-T17 |
| §7 Smoke checklist | T19 (README) |

### Execution choice

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
