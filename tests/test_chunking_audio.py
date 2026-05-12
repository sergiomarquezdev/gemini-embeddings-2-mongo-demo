import subprocess
import tempfile
from pathlib import Path
from chunking import chunk_audio, AudioChunk


FIXTURE = Path("tests/fixtures/sample_300s.wav")


def _probe_duration_seconds(data: bytes) -> float:
    """Probe duration via temp file to avoid WAV pipe header issues on Windows."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", "-i", tmp_path],
            capture_output=True, check=True,
        )
        return float(p.stdout.decode().strip())
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_300s_audio_splits_into_2_chunks():
    """300s audio with max=170s and overlap=10s -> [0-170], [160-300]."""
    chunks = chunk_audio(FIXTURE.read_bytes(), max_seconds=170, overlap_seconds=10)
    assert len(chunks) == 2
    assert chunks[0].time_start == 0
    assert chunks[0].time_end == 170
    assert chunks[1].time_start == 160
    assert chunks[1].time_end == 300
    assert all(c.n_total == 2 for c in chunks)


def test_short_audio_returns_single_chunk():
    """120s audio (<= 180s hard limit) -> single chunk passthrough."""
    short = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=120",
         "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    chunks = chunk_audio(short, max_seconds=170, overlap_seconds=10)
    assert len(chunks) == 1


def test_each_chunk_audio_is_valid():
    chunks = chunk_audio(FIXTURE.read_bytes(), max_seconds=170, overlap_seconds=10)
    for c in chunks:
        d = _probe_duration_seconds(c.audio_bytes)
        # ffmpeg has +/- 0.5s slack on cuts; assert within tolerance
        assert abs(d - (c.time_end - c.time_start)) < 1.0
