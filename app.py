"""FastAPI demo: gemini-embedding-2 + MongoDB Atlas Local Vector Search.

Single-file monolith for didactic clarity. Helpers in chunking.py and archives.py.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import filetype

from dotenv import load_dotenv
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, DuplicateKeyError


def _is_dup_key_bulk(exc: BulkWriteError) -> bool:
    """True iff every write error in the bulk failure is a duplicate key (code 11000)."""
    errors = exc.details.get("writeErrors", []) if exc.details else []
    return bool(errors) and all(e.get("code") == 11000 for e in errors)

from chunking import chunk_text, chunk_pdf, chunk_audio, chunk_video, _audio_duration_seconds
from archives import extract_archive, ArchiveError, ArchiveEntry
from mongo_setup import healthcheck, init_indexes
from vertex_client import VertexClient

load_dotenv()
logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Helpers: MIME sniffing, hashing, modality dispatch
# ---------------------------------------------------------------------------

MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"
MODALITY_PDF = "pdf"
MODALITY_AUDIO = "audio"
MODALITY_VIDEO = "video"

SUPPORTED_AUDIO = {"audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav"}
SUPPORTED_VIDEO = {"video/mp4", "video/mpeg"}
SUPPORTED_IMAGE = {"image/png", "image/jpeg", "image/webp", "image/bmp",
                   "image/heic", "image/heif", "image/avif"}
SUPPORTED_PDF = {"application/pdf"}
SUPPORTED_TEXT_EXT = {".txt", ".md"}
SUPPORTED_ARCHIVE = {"application/zip", "application/x-zip-compressed",
                     "application/vnd.rar", "application/x-rar", "application/x-rar-compressed"}


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


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

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
    except BulkWriteError as exc:
        if _is_dup_key_bulk(exc):
            # race: another request inserted the same chunks — caller resolves via _existing_doc
            raise DuplicateKeyError(exc.details)
        raise
    except DuplicateKeyError:
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


def _ingest_extracted(db, *, vertex, entry: ArchiveEntry, archive_filename: str) -> dict:
    """Process a single file from inside an archive. Returns a per-file result dict."""
    if entry.skipped:
        return {"filename": entry.name, "status": "skipped", "reason": entry.skip_reason}
    raw = entry.data
    mime = sniff_mime(raw, fallback_name=entry.name)
    modality = modality_of(mime)
    if modality is None:
        return {"filename": entry.name, "status": "skipped", "reason": f"unsupported mime: {mime}"}
    assert mime is not None  # narrowed by modality_of check above
    ch = content_hash(raw)
    existing = _existing_doc(db, ch)
    if existing:
        return {"filename": entry.name, "doc_id": existing["parent_doc_id"], "status": "already_indexed"}
    _, storage_path = _save_uploaded_file(raw, entry.name)
    try:
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
        db[MONGO_COLLECTION].update_many(
            {"parent_doc_id": res["doc_id"]},
            {"$set": {"source_archive": {"filename": archive_filename, "entry_name": entry.name}}},
        )
        return {"filename": entry.name, "doc_id": res["doc_id"], "n_chunks": res["n_chunks"], "status": "ok"}
    except Exception as exc:
        logger.exception("archive entry %s failed", entry.name)
        return {"filename": entry.name, "status": "failed",
                "reason": f"{type(exc).__name__}"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if app.state.vertex is None:
        raise HTTPException(status_code=503, detail={"error": "Vertex AI not configured (set GCP_PROJECT)"})
    raw = file.file.read()
    _ensure_within_size(raw)
    mime = sniff_mime(raw, fallback_name=file.filename or "")

    if mime in SUPPORTED_ARCHIVE:
        try:
            entries = extract_archive(raw, mime_type=mime,
                                      max_files=MAX_ARCHIVE_FILES,
                                      max_uncompressed_mb=MAX_ARCHIVE_UNCOMPRESSED_MB)
        except ArchiveError as e:
            raise HTTPException(status_code=422, detail={"error": str(e)})
        results = [_ingest_extracted(app.state.db, vertex=app.state.vertex,
                                     entry=e, archive_filename=file.filename or "")
                   for e in entries]
        summary = {"total": len(results),
                   "ok": sum(1 for r in results if r["status"] == "ok"),
                   "already_indexed": sum(1 for r in results if r["status"] == "already_indexed"),
                   "skipped": sum(1 for r in results if r["status"] == "skipped"),
                   "failed": sum(1 for r in results if r["status"] == "failed")}
        return {"archive": file.filename, "extracted": results, "summary": summary}

    modality = modality_of(mime)
    if modality is None:
        raise HTTPException(status_code=415,
                            detail={"error": "unsupported mime",
                                    "detected_mime": mime,
                                    "supported": sorted(SUPPORTED_IMAGE | SUPPORTED_PDF |
                                                        SUPPORTED_AUDIO | SUPPORTED_VIDEO |
                                                        {"text/plain"})})
    assert mime is not None  # narrowed by modality_of check above

    ch = content_hash(raw)
    existing = _existing_doc(app.state.db, ch)
    if existing:
        return {"doc_id": existing["parent_doc_id"], "n_chunks": existing["n_chunks_total"],
                "modality": existing["modality"], "status": "already_indexed",
                "indexed_at": existing["created_at"].isoformat()}

    _, storage_path = _save_uploaded_file(raw, file.filename or "unnamed")

    try:
        kwargs = dict(raw=raw, filename=file.filename or "", mime=mime,
                      content_h=ch, storage_path=storage_path, vertex=app.state.vertex)
        if modality == MODALITY_TEXT:
            return _ingest_text(app.state.db, raw_bytes=raw, filename=file.filename or "",
                                mime_type=mime, content_h=ch, storage_path=storage_path,
                                vertex=app.state.vertex)
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


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_VALID_MODALITIES = {"text", "image", "pdf", "audio", "video"}


class SearchPayload(BaseModel):
    query: Optional[str] = None
    modality: Optional[list[str]] = None
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("modality")
    @classmethod
    def _check_modality(cls, v):
        if v is None:
            return v
        bad = [m for m in v if m not in _VALID_MODALITIES]
        if bad:
            raise ValueError(f"unknown modality values: {bad}")
        return v


def _vector_search(query_vector: list[float], *, modality_filter, limit: int) -> dict:
    vfilter: dict = {"status": "ok"}
    if modality_filter:
        vfilter["modality"] = {"$in": modality_filter}
    pipeline = [
        {"$vectorSearch": {
            "index": "vector_index", "path": "vector", "queryVector": query_vector,
            "numCandidates": limit * 20, "limit": limit * 2, "filter": vfilter,
        }},
        {"$project": {"score": {"$meta": "vectorSearchScore"},
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
    try:
        out = list(app.state.db[MONGO_COLLECTION].aggregate(pipeline))
    except Exception as exc:
        logger.warning("vector search failed (likely INITIAL_SYNC): %s", exc)
        out = []
    return {"results": [{
        "doc_id": r["_id"],
        "score": r["best_score"],
        "matched_chunks": r["matched_chunks"],
        "best_chunk": {k: v for k, v in r["best_chunk"].items() if k != "_id"},
    } for r in out]}


@app.post("/search")
def search(payload: SearchPayload = Body(...)):
    if app.state.vertex is None:
        raise HTTPException(status_code=503, detail={"error": "Vertex AI not configured (set GCP_PROJECT)"})
    if not payload.query:
        raise HTTPException(status_code=400, detail={"error": "provide query or file"})
    emb = app.state.vertex.embed_query(text=payload.query)
    return _vector_search(emb.vector, modality_filter=payload.modality, limit=payload.limit)


@app.post("/search/file")
def search_file(file: UploadFile = File(...)):
    if app.state.vertex is None:
        raise HTTPException(status_code=503, detail={"error": "Vertex AI not configured (set GCP_PROJECT)"})
    raw = file.file.read()
    _ensure_within_size(raw)
    mime = sniff_mime(raw, fallback_name=file.filename or "")
    if modality_of(mime) is None:
        raise HTTPException(status_code=415, detail={"error": "unsupported mime"})
    emb = app.state.vertex.embed_query(file_bytes=raw, mime_type=mime)
    return _vector_search(emb.vector, modality_filter=None, limit=10)


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/files/{doc_id}")
def serve_file(doc_id: str):
    # Validate doc_id: only alnum + hyphen, max 64 chars (rejects path traversal at route level)
    if not all(c.isalnum() or c == "-" for c in doc_id) or len(doc_id) > 64:
        raise HTTPException(status_code=400, detail={"error": "invalid doc_id"})
    doc = app.state.db[MONGO_COLLECTION].find_one(
        {"parent_doc_id": doc_id},
        projection={"storage_path": 1, "filename": 1, "mime_type": 1},
    )
    if not doc:
        raise HTTPException(status_code=404, detail={"error": "doc not found"})
    p = Path(doc["storage_path"]).resolve()
    uploads_real = UPLOADS_DIR.resolve()
    # Defense in depth: ensure resolved path is under uploads/ (handles uploads_evil prefix bypass)
    if not p.is_relative_to(uploads_real):
        raise HTTPException(status_code=400, detail={"error": "invalid storage_path"})
    if not p.exists():
        raise HTTPException(status_code=404, detail={"error": "file missing on disk"})
    return FileResponse(
        path=str(p),
        media_type=doc["mime_type"],
        filename=doc["filename"],
        headers={"Content-Disposition": f'attachment; filename="{doc["filename"]}"'},
    )
