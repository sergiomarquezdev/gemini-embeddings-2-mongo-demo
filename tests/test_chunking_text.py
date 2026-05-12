from chunking import chunk_text, TextChunk


def make_token_counter():
    """Returns a callable that simulates count_tokens with ~1.5 tokens per word."""

    def _count(text: str) -> int:
        return max(1, int(len(text.split()) * 1.5))

    return _count


def test_short_text_returns_single_chunk():
    text = "Hello world. " * 100  # ~150 tokens
    counter = make_token_counter()
    chunks = chunk_text(text, count_tokens=counter, max_tokens=7000, overlap_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].n_total == 1


def test_long_text_splits_into_multiple_chunks():
    # 10000 words ~ 15000 tokens — should split into ~3 chunks
    text = " ".join(["palabra"] * 10_000)
    counter = make_token_counter()
    chunks = chunk_text(text, count_tokens=counter, max_tokens=7000, overlap_tokens=500)
    assert len(chunks) >= 2
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.n_total == len(chunks) for c in chunks)
    # chunk_index sequential
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_have_overlap():
    """With overlap_tokens=200 and ~1.5 tokens/word, expect ~133 words of overlap.
    Use a 200-word window on each side to reliably detect the overlap."""
    text = " ".join([f"w{i}" for i in range(5000)])
    counter = make_token_counter()
    chunks = chunk_text(text, count_tokens=counter, max_tokens=2000, overlap_tokens=200)
    assert len(chunks) >= 2
    end_of_first = chunks[0].text.split()[-200:]
    start_of_second = chunks[1].text.split()[:200]
    overlap = set(end_of_first) & set(start_of_second)
    assert len(overlap) > 0, (
        f"no overlap detected (chunk0 ends {end_of_first[-3:]}, "
        f"chunk1 starts {start_of_second[:3]})"
    )


def test_empty_text_returns_empty_list():
    counter = make_token_counter()
    assert chunk_text("", count_tokens=counter, max_tokens=7000, overlap_tokens=500) == []


def test_exactly_max_tokens_does_not_split():
    # Build a text whose token count equals exactly max_tokens
    text = " ".join(["word"] * 4666)  # 4666 * 1.5 = 6999 tokens
    counter = make_token_counter()
    chunks = chunk_text(text, count_tokens=counter, max_tokens=7000, overlap_tokens=500)
    assert len(chunks) == 1
