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
