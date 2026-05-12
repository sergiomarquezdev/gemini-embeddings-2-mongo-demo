import io
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from chunking import chunk_pdf, PdfChunk

FIXTURE = Path("tests/fixtures/sample_15p.pdf")


def test_15_page_pdf_splits_into_5_chunks_with_overlap():
    """15 pages, max=4, overlap=1 -> chunks pages [1-4], [4-7], [7-10], [10-13], [13-15]."""
    chunks = chunk_pdf(FIXTURE.read_bytes(), max_pages=4, overlap_pages=1)
    assert all(isinstance(c, PdfChunk) for c in chunks)
    page_ranges = [(c.page_start, c.page_end) for c in chunks]
    assert page_ranges == [(1, 4), (4, 7), (7, 10), (10, 13), (13, 15)]
    assert all(c.n_total == 5 for c in chunks)


def test_5_page_pdf_returns_single_chunk():
    """5 pages with max=4: still under hard limit (6) so single chunk passthrough."""
    w = PdfWriter()
    for _ in range(5):
        w.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    w.write(buf)
    chunks = chunk_pdf(buf.getvalue(), max_pages=4, overlap_pages=1)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 5
    assert chunks[0].chunk_index == 0


def test_each_chunk_bytes_is_valid_pdf():
    """Each chunk's bytes must parse as a valid PDF on its own."""
    chunks = chunk_pdf(FIXTURE.read_bytes(), max_pages=4, overlap_pages=1)
    for c in chunks:
        r = PdfReader(io.BytesIO(c.pdf_bytes))
        assert len(r.pages) == c.page_end - c.page_start + 1
