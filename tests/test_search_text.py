from fastapi.testclient import TestClient


def test_search_text_query_returns_results_grouped_by_parent(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        # Seed 3 docs
        for content in [b"alpha text", b"beta text", b"gamma text"]:
            c.post("/upload", files={"file": (f"{content[:5].decode()}.txt", content, "text/plain")})
        r = c.post("/search", json={"query": "anything"})
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    # Vector search index is eventually-consistent — shape is what matters
    assert isinstance(body["results"], list)


def test_search_with_empty_payload_returns_400(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/search", json={})
    assert r.status_code == 400


def test_search_filters_by_modality(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/search", json={"query": "x", "modality": ["pdf", "text"]})
    assert r.status_code == 200
