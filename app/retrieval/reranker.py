"""Cross-encoder reranking: re-scores (query, chunk) pairs with a model that reads both together."""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app import config
from app.retrieval.hybrid import search_hybrid


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(config.RERANKER_MODEL)


def rerank(query: str, chunks: list[dict], k: int = config.TOP_K) -> list[dict]:
    """Return the k best chunks by cross-encoder score. The input list is not modified."""
    if not chunks:
        return []
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = get_reranker().predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
    results = []
    for chunk, score in ranked[:k]:
        item = dict(chunk)
        item["score"] = float(score)
        results.append(item)
    return results


def search_rerank(
    query: str,
    k: int = config.TOP_K,
    candidates: int = config.RERANK_CANDIDATES,
) -> list[dict]:
    """Take hybrid's top `candidates`, rerank them, return the top k."""
    return rerank(query, search_hybrid(query, k=candidates), k=k)