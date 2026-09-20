"""Keyword (BM25) retrieval using Elasticsearch full-text search on the `text` field.

Elasticsearch scores text matches with BM25 by default, so no re-ingestion is needed:
the `text` field was already indexed as full text in Version 1.
"""
from app import config
from app.es_client import get_es_client


def search_bm25(query: str, k: int = config.TOP_K) -> list[dict]:
    """Return the top-k chunks by BM25 score. Each hit is a dict with a 'score' key added."""
    es = get_es_client()
    response = es.search(
        index=config.ES_INDEX,
        query={"match": {"text": {"query": query}}},
        source=["chunk_id", "source", "page", "chunk_index", "text"],
        size=k,
    )
    hits = []
    for hit in response["hits"]["hits"]:
        item = dict(hit["_source"])
        item["score"] = hit["_score"]
        hits.append(item)
    return hits