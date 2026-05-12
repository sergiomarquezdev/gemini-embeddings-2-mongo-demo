from fastapi.testclient import TestClient


def test_files_endpoint_returns_attachment(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/upload", files={"file": ("hello.txt", b"hello", "text/plain")})
        doc_id = r.json()["doc_id"]
        r2 = c.get(f"/files/{doc_id}")
    assert r2.status_code == 200
    assert r2.headers["content-disposition"].startswith("attachment;")


def test_files_endpoint_404_for_unknown(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.get("/files/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_files_endpoint_rejects_path_traversal(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.get("/files/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code in (400, 404, 422)
