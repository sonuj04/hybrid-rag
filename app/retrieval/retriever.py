"""Single entry point for retrieval: pick a mode by name."""
from app import config
from app.retrieval.bm25 import search_bm25
from app.retrieval.dense import search_dense
from app.retrieval.hybrid import search_hybrid
from app.retrieval.reranker import search_rerank

MODES = ["dense", "bm25", "hybrid", "rerank"]


def search(query: str, mode: str = "hybrid", k: int = config.TOP_K) -> list[dict]:
    if mode == "dense":
        return search_dense(query, k=k)
    if mode == "bm25":
        return search_bm25(query, k=k)
    if mode == "hybrid":
        return search_hybrid(query, k=k)
    if mode == "rerank":
        return search_rerank(query, k=k)
    raise ValueError(f"Unknown retrieval mode '{mode}'. Choose from: {MODES}")