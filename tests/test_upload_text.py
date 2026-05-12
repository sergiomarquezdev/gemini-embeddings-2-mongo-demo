from fastapi.testclient import TestClient


def test_upload_txt_creates_one_chunk(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-2")
    import app as app_module
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/upload", files={"file": ("hello.txt", b"Hello world", "text/plain")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_chunks"] == 1
    assert body["modality"] == "text"
    assert "doc_id" in body


def test_upload_same_file_twice_returns_already_indexed(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-2")
    import app as app_module
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r1 = c.post("/upload", files={"file": ("a.txt", b"same content", "text/plain")})
        assert r1.status_code == 200
        r2 = c.post("/upload", files={"file": ("a.txt", b"same content", "text/plain")})
        assert r2.status_code == 200
        assert r2.json()["status"] == "already_indexed"


def test_upload_unsupported_mime_returns_415(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    monkeypatch.setenv("EMBEDDING_DIM", "1536")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-2")
    import app as app_module
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        # MZ = Windows exe magic
        r = c.post("/upload", files={"file": ("evil.bin", b"MZ\x90\x00" + b"\x00" * 100, "application/octet-stream")})
    assert r.status_code == 415
