from unittest.mock import MagicMock
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
    result = vc.embed_doc(file_bytes=b"%PDF-1.7", mime_type="application/pdf")
    cfg = mock_genai_client.models.embed_content.call_args.kwargs["config"]
    assert cfg.document_ocr is True
    assert result.flags == {"document_ocr": True}


def test_embed_doc_video_sets_audio_track_extraction_flag(mock_genai_client):
    vc = VertexClient(genai_client=mock_genai_client, model="gemini-embedding-2", dim=1536)
    result = vc.embed_doc(file_bytes=b"\x00\x00\x00\x20ftypmp42", mime_type="video/mp4")
    cfg = mock_genai_client.models.embed_content.call_args.kwargs["config"]
    assert cfg.audio_track_extraction is True
    assert result.flags == {"audio_track_extraction": True}


def test_count_tokens_returns_int(mock_genai_client):
    vc = VertexClient(genai_client=mock_genai_client, model="gemini-embedding-2", dim=1536)
    n = vc.count_tokens("some text")
    assert n == 42
