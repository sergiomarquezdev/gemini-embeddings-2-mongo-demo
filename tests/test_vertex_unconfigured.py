"""Regression: /upload and /search must 503 when Vertex AI is not configured."""
from fastapi.testclient import TestClient
import app as app_module


def test_upload_returns_503_when_vertex_missing(monkeypatch, test_db_name, test_db):
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = None
        r = c.post("/upload", files={"file": ("hello.txt", b"hi", "text/plain")})
    assert r.status_code == 503
    assert "Vertex" in r.json()["detail"]["error"]


def test_search_returns_503_when_vertex_missing(monkeypatch, test_db_name, test_db):
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = None
        r = c.post("/search", json={"query": "anything"})
    assert r.status_code == 503


def test_search_file_returns_503_when_vertex_missing(monkeypatch, test_db_name, test_db):
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = None
        r = c.post("/search/file", files={"file": ("a.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "image/png")})
    assert r.status_code == 503
