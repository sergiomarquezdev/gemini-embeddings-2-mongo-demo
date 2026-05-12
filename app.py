"""FastAPI demo: gemini-embedding-2 + MongoDB Atlas Local Vector Search.

Single-file monolith for didactic clarity. Helpers in chunking.py and archives.py.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import filetype

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from chunking import chunk_text
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


# ---------------------------------------------------------------------------
# Helpers: MIME sniffing, hashing, modality dispatch
# ---------------------------------------------------------------------------

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
    except DuplicateKeyError:
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
