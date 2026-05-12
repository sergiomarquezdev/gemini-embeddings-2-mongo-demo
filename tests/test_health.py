from fastapi.testclient import TestClient


def test_health_returns_status(monkeypatch, test_db_name):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017/?directConnection=true")
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-2")
    from app import app
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mongo"] in ("ok", "down")
    assert "vector_index" in body
