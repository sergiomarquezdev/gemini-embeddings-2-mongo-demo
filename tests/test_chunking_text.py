from unittest.mock import MagicMock
from chunking import chunk_text, TextChunk


def make_token_counter(tokens_per_chunk):
    """Returns a callable that simulates count_tokens.

    `tokens_per_chunk` maps text-prefix-length -> token count.
    For tests we use a simple word-based heuristic: ~1.5 tokens per word.
    """

    def _count(text: str) -> int:
        return max(1, int(len(text.split()) * 1.5))

    return _count


def test_short_text_returns_single_chunk():
    text = "Hello world. " * 100  # ~150 tokens
    counter = make_token_counter(None)
    chunks = chunk_text(text, count_tokens=counter, max_tokens=7000, overlap_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].n_total == 1


def test_long_text_splits_into_multiple_chunks():
    # 10000 words ~ 15000 tokens — should split into ~3 chunks
    text = " ".join(["palabra"] * 10_000)
    counter = make_token_counter(None)
    chunks = chunk_text(text, count_tokens=counter, max_tokens=7000, overlap_tokens=500)
    assert len(chunks) >= 2
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.n_total == len(chunks) for c in chunks)
    # chunk_index sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_have_overlap():
    text = " ".join([f"w{i}" for i in range(5000)])
    counter = make_token_counter(None)
    chunks = chunk_text(text, count_tokens=counter, max_tokens=2000, overlap_tokens=200)
    assert len(chunks) >= 2
    # The last words of chunk N should appear at the start of chunk N+1
    end_of_first = chunks[0].text.split()[-50:]
    start_of_second = chunks[1].text.split()[:50]
    overlap = set(end_of_first) & set(start_of_second)
    assert len(overlap) > 0, "expected token overlap between consecutive chunks"


def test_empty_text_returns_empty_list():
    counter = make_token_counter(None)
    assert chunk_text("", count_tokens=counter, max_tokens=7000, overlap_tokens=500) == []


def test_exactly_max_tokens_does_not_split():
    # Build a text whose token count equals exactly max_tokens
    text = " ".join(["word"] * 4666)  # 4666 * 1.5 = 6999 tokens
    counter = make_token_counter(None)
    chunks = chunk_text(text, count_tokens=counter, max_tokens=7000, overlap_tokens=500)
    assert len(chunks) == 1
