"""FastAPI demo: gemini-embedding-2 + MongoDB Atlas Local Vector Search.

Single-file monolith for didactic clarity. Helpers in chunking.py and archives.py.
"""
from __future__ import annotations

import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path

import filetype

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
