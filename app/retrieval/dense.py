"""Dense (embedding-based) retrieval using Elasticsearch kNN search."""
from app import config
from app.embeddings import embed_query
from app.es_client import get_es_client


def search_dense(query: str, k: int = config.TOP_K, num_candidates: int = 100) -> list[dict]:
    """Return the top-k chunks for a query. Each hit is a dict with a 'score' key added.

    For cosine similarity, Elasticsearch's score is (1 + cosine) / 2, so it lies between 0 and 1.
    """
    es = get_es_client()
    response = es.search(
        index=config.ES_INDEX,
        knn={
            "field": "embedding",
            "query_vector": embed_query(query),
            "k": k,
            "num_candidates": num_candidates,
        },
        source=["chunk_id", "source", "page", "chunk_index", "text"],
        size=k,
    )
    hits = []
    for hit in response["hits"]["hits"]:
        item = dict(hit["_source"])
        item["score"] = hit["_score"]
        hits.append(item)
    return hits