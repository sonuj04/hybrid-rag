import pytest

from app.ingestion.chunker import split_text


def test_short_text_is_a_single_chunk():
    assert split_text("hello world", chunk_size=100, overlap=20) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert split_text("   ", chunk_size=100, overlap=20) == []


def test_chunks_never_exceed_chunk_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_consecutive_chunks_overlap_and_nothing_is_lost():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = split_text(text, chunk_size=100, overlap=20)
    # the last word of each chunk appears again in the next chunk
    for current, following in zip(chunks, chunks[1:]):
        assert current.split()[-1] in following.split()
    # every original word appears in at least one chunk
    assert set(text.split()) == set(" ".join(chunks).split())


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=50, overlap=50)