"""Happy-path test for POST /search/file (multimodal query by file)."""
from fastapi.testclient import TestClient
import app as app_module


def test_search_file_with_png_returns_results_shape(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/search/file", files={"file": ("query.png", png, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "results" in body
    assert isinstance(body["results"], list)


def test_search_file_with_unsupported_mime_returns_415(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/search/file", files={"file": ("evil.bin", b"MZ\x90\x00" + b"\x00" * 100, "application/octet-stream")})
    assert r.status_code == 415
