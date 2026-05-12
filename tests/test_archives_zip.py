import io
import zipfile
import pytest
from archives import extract_archive, ArchiveEntry, ArchiveError


def _make_zip(entries: dict[str, bytes], compress=zipfile.ZIP_STORED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_extract_basic_zip_returns_entries():
    z = _make_zip({"a.txt": b"hello", "b.png": b"\x89PNG\r\n"})
    entries = extract_archive(z, mime_type="application/zip", max_files=10, max_uncompressed_mb=50)
    names = sorted(e.name for e in entries)
    assert names == ["a.txt", "b.png"]
    assert all(isinstance(e, ArchiveEntry) for e in entries)


def test_zip_bomb_rejected_pre_extraction():
    """Headers declare >50MB total — must reject WITHOUT extracting."""
    huge = b"x" * (10 * 1024 * 1024)  # 10MB
    z = _make_zip({f"f{i}.bin": huge for i in range(6)})  # declared 60MB total
    with pytest.raises(ArchiveError, match="exceeds.*uncompressed"):
        extract_archive(z, mime_type="application/zip", max_files=100, max_uncompressed_mb=50)


def test_too_many_files_rejected():
    z = _make_zip({f"f{i}.txt": b"x" for i in range(15)})
    with pytest.raises(ArchiveError, match="too many"):
        extract_archive(z, mime_type="application/zip", max_files=10, max_uncompressed_mb=50)


def test_path_traversal_entries_skipped():
    """Entries with `../` or absolute paths must be rejected."""
    z = _make_zip({"../etc/passwd": b"root:x:0", "good.txt": b"safe"})
    entries = extract_archive(z, mime_type="application/zip", max_files=10, max_uncompressed_mb=50)
    names = [e.name for e in entries]
    assert names == ["good.txt"]


def test_encrypted_zip_rejected():
    import struct
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zi = zipfile.ZipInfo("secret.txt")
        zi.flag_bits |= 0x1  # encrypted flag (stripped by writestr; patched below)
        zf.writestr(zi, b"secret")
    # Python's zipfile strips flag_bits during write; patch the central directory
    # entry directly so the encrypted bit (0x1) is preserved on re-open.
    data = bytearray(buf.getvalue())
    cd_pos = data.find(b"PK\x01\x02")
    struct.pack_into("<H", data, cd_pos + 8, 0x1)
    with pytest.raises(ArchiveError, match="encrypted"):
        extract_archive(bytes(data), mime_type="application/zip", max_files=10, max_uncompressed_mb=50)


def test_nested_archive_skipped_with_warning():
    inner = _make_zip({"inside.txt": b"x"})
    z = _make_zip({"nested.zip": inner, "ok.txt": b"y"})
    entries = extract_archive(z, mime_type="application/zip", max_files=10, max_uncompressed_mb=50)
    names = [e.name for e in entries]
    # nested.zip is skipped (depth=1 policy)
    assert "ok.txt" in names
    skipped = [e for e in entries if e.skipped]
    assert any(e.name == "nested.zip" for e in skipped)
