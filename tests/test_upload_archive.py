import io, zipfile
from fastapi.testclient import TestClient


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for k, v in entries.items():
            zf.writestr(k, v)
    return buf.getvalue()


def test_upload_zip_with_mix_returns_per_file_summary(monkeypatch, test_db_name, test_db, mock_vertex_client):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    z = _zip({"a.txt": b"hello", "b.png": png, "evil.exe": b"MZ\x90\x00" + b"\x00" * 100})
    monkeypatch.setenv("MONGO_DB", test_db_name)
    import app as app_module
    monkeypatch.setattr(app_module, "MONGO_DB", test_db_name)
    with TestClient(app_module.app) as c:
        app_module.app.state.vertex = mock_vertex_client
        r = c.post("/upload", files={"file": ("mix.zip", z, "application/zip")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "extracted" in body
    statuses = {e["filename"]: e["status"] for e in body["extracted"]}
    assert statuses["a.txt"] == "ok"
    assert statuses["b.png"] == "ok"
    assert statuses["evil.exe"] == "skipped"
    assert body["summary"]["ok"] == 2
    assert body["summary"]["skipped"] == 1
