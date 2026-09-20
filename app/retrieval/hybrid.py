"""Hybrid retrieval: merge dense and BM25 results with Reciprocal Rank Fusion (RRF)."""
from app import config
from app.retrieval.bm25 import search_bm25
from app.retrieval.dense import search_dense

RRF_K = 60  # standard RRF constant


def reciprocal_rank_fusion(result_lists: dict[str, list[dict]], rrf_k: int = RRF_K) -> list[dict]:
    """Merge several ranked lists of chunks into one ranked list.

    `result_lists` maps a name (such as "dense" or "bm25") to a list of chunk dicts, best first.
    Every chunk earns 1 / (rrf_k + rank) from each list it appears in (rank starts at 1),
    and the points are summed. Raw scores from the individual retrievers are ignored.

    Each returned chunk has 'score' (the RRF score) and 'ranks' (its rank in each list).
    """
    fused: dict[str, dict] = {}
    for name, hits in result_lists.items():
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {**hit, "score": 0.0, "ranks": {}}
            fused[chunk_id]["score"] += 1.0 / (rrf_k + rank)
            fused[chunk_id]["ranks"][name] = rank
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)


def search_hybrid(
    query: str,
    k: int = config.TOP_K,
    candidates: int = config.HYBRID_CANDIDATES,
) -> list[dict]:
    """Take the top `candidates` from dense and BM25, merge with RRF, return the top k."""
    dense_hits = search_dense(query, k=candidates)
    bm25_hits = search_bm25(query, k=candidates)
    fused = reciprocal_rank_fusion({"dense": dense_hits, "bm25": bm25_hits})
    return fused[:k]