import pytest
from app import sniff_mime, content_hash, modality_of, MODALITY_TEXT, MODALITY_IMAGE, MODALITY_PDF


def test_content_hash_is_sha256_prefixed():
    h = content_hash(b"hello")
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def test_sniff_mime_detects_png_from_bytes():
    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    assert sniff_mime(png_magic, fallback_name="x.png") == "image/png"


def test_sniff_mime_rejects_exe_renamed_to_pdf():
    """A binary that's actually MZ-prefixed (Windows exe) is NOT pdf."""
    exe = b"MZ\x90\x00" + b"\x00" * 100
    detected = sniff_mime(exe, fallback_name="evil.pdf")
    assert detected != "application/pdf"


def test_modality_of_maps_pdf():
    assert modality_of("application/pdf") == MODALITY_PDF


def test_modality_of_maps_image_variants():
    for m in ("image/png", "image/jpeg", "image/webp", "image/heic"):
        assert modality_of(m) == MODALITY_IMAGE


def test_modality_of_maps_audio_mp3_and_mpeg():
    assert modality_of("audio/mp3") == "audio"
    assert modality_of("audio/mpeg") == "audio"


def test_modality_of_unknown_returns_none():
    assert modality_of("application/x-msdownload") is None
