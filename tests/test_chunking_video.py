import subprocess
import tempfile
from pathlib import Path

import pytest

from chunking import chunk_video, VideoChunk


FIXTURE = Path("tests/fixtures/sample_200s.mp4")


def _probe_duration_seconds(data: bytes) -> float:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        tmp = Path(f.name)
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(tmp)],
            capture_output=True, check=True,
        )
        return float(p.stdout.decode().strip())
    finally:
        tmp.unlink(missing_ok=True)


def test_200s_video_splits_into_chunks_of_70s():
    """200s with max=70s and overlap=10s -> [0-70], [60-130], [120-190], [180-200]."""
    chunks = chunk_video(FIXTURE.read_bytes(), max_seconds=70, overlap_seconds=10)
    assert len(chunks) == 4
    starts_ends = [(c.time_start, c.time_end) for c in chunks]
    assert starts_ends[0] == (0, 70)
    assert starts_ends[-1][1] == 200
    assert all(c.n_total == 4 for c in chunks)


def test_short_video_returns_single_chunk():
    """30s video (<= 81s hard limit) -> single chunk passthrough."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_in = Path(f.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=30:size=320x240:rate=1",
             "-c:v", "libx264", str(tmp_in)],
            capture_output=True, check=True,
        )
        short = tmp_in.read_bytes()
    finally:
        tmp_in.unlink(missing_ok=True)
    chunks = chunk_video(short, max_seconds=70, overlap_seconds=10)
    assert len(chunks) == 1


def test_each_chunk_video_is_valid_duration():
    chunks = chunk_video(FIXTURE.read_bytes(), max_seconds=70, overlap_seconds=10)
    for c in chunks:
        d = _probe_duration_seconds(c.video_bytes)
        assert abs(d - (c.time_end - c.time_start)) < 2.0
