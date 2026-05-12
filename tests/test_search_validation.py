"""Validation: SearchPayload rejects bad limit and modality values with 422."""
from fastapi.testclient import TestClient
import app as app_module


def test_search_with_limit_zero_returns_422(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/search", json={"query": "x", "limit": 0})
    assert r.status_code == 422


def test_search_with_limit_too_large_returns_422(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/search", json={"query": "x", "limit": 99999})
    assert r.status_code == 422


def test_search_with_unknown_modality_returns_422(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/search", json={"query": "x", "modality": ["text", "spaceship"]})
    assert r.status_code == 422
