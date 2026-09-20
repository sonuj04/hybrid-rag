import pytest

from app.retrieval.hybrid import RRF_K, reciprocal_rank_fusion


def make_hit(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "source": "a.pdf", "page": 1, "text": chunk_id, "score": 0.0}


def test_chunks_found_by_both_retrievers_rank_first():
    dense = [make_hit("a"), make_hit("b"), make_hit("c")]
    bm25 = [make_hit("a"), make_hit("d"), make_hit("c")]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25})
    assert [hit["chunk_id"] for hit in fused][:2] == ["a", "c"]


def test_score_of_a_single_rank_one_hit():
    fused = reciprocal_rank_fusion({"dense": [make_hit("a")]})
    assert fused[0]["score"] == pytest.approx(1 / (RRF_K + 1))


def test_ranks_are_recorded_per_retriever():
    dense = [make_hit("a"), make_hit("b")]
    bm25 = [make_hit("b")]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25})
    by_id = {hit["chunk_id"]: hit for hit in fused}
    assert by_id["a"]["ranks"] == {"dense": 1}
    assert by_id["b"]["ranks"] == {"dense": 2, "bm25": 1}


def test_empty_input_gives_empty_output():
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []