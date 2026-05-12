import shutil
from pathlib import Path
import pytest
from archives import extract_archive, ArchiveError

FIXTURE = Path("tests/fixtures/sample.rar")


@pytest.mark.skipif(
    not FIXTURE.exists() or shutil.which("unrar") is None,
    reason="sample.rar fixture missing or unrar binary not on PATH",
)
def test_extract_basic_rar():
    entries = extract_archive(
        FIXTURE.read_bytes(),
        mime_type="application/vnd.rar",
        max_files=10,
        max_uncompressed_mb=50,
    )
    names = sorted(e.name.split("/")[-1] for e in entries if not e.skipped)
    assert "a.txt" in names and "b.txt" in names


def test_unsupported_archive_mime_raises():
    with pytest.raises(ArchiveError, match="unsupported archive mime"):
        extract_archive(b"x", mime_type="application/x-7z-compressed",
                        max_files=10, max_uncompressed_mb=50)
