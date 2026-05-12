from pathlib import Path
from fastapi.testclient import TestClient


def test_upload_png_image_creates_single_chunk(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
        r = c.post("/upload", files={"file": ("a.png", png, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["modality"] == "image"
    assert body["n_chunks"] == 1


def test_upload_pdf_creates_chunks_per_page_block(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    fixture = Path("tests/fixtures/sample_15p.pdf").read_bytes()
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/upload", files={"file": ("doc.pdf", fixture, "application/pdf")})
    assert r.status_code == 200
    assert r.json()["n_chunks"] == 5  # 15 pages / 4 with 1 overlap


def test_upload_audio_creates_chunks(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    fixture = Path("tests/fixtures/sample_300s.wav").read_bytes()
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/upload", files={"file": ("a.wav", fixture, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["modality"] == "audio"
    assert body["n_chunks"] == 2


def test_upload_video_above_cost_cap_returns_413(monkeypatch, test_db_name, test_db, mock_vertex_client):
    monkeypatch.setenv("MONGO_DB", test_db_name)
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    monkeypatch.setattr(app_module, "MAX_TOTAL_EMBED_SECONDS", 10)  # absurdly low
    fixture = Path("tests/fixtures/sample_200s.mp4").read_bytes()
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/upload", files={"file": ("v.mp4", fixture, "video/mp4")})
    assert r.status_code == 413
