"""Creates the Elasticsearch index and writes chunks (with their embeddings) into it."""
from elasticsearch import Elasticsearch, helpers

from app import config
from app.ingestion.chunker import Chunk


def create_index_if_missing(es: Elasticsearch) -> None:
    if es.indices.exists(index=config.ES_INDEX):
        return
    es.indices.create(
        index=config.ES_INDEX,
        settings={"number_of_shards": 1, "number_of_replicas": 0},
        mappings={
            "properties": {
                "chunk_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "page": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "text": {"type": "text"},  # also used for BM25 in Version 2
                "embedding": {
                    "type": "dense_vector",
                    "dims": config.EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    )


def delete_index(es: Elasticsearch) -> None:
    if es.indices.exists(index=config.ES_INDEX):
        es.indices.delete(index=config.ES_INDEX)


def delete_source(es: Elasticsearch, source: str) -> None:
    """Remove all chunks that came from one file (so re-ingesting never leaves stale chunks)."""
    es.delete_by_query(
        index=config.ES_INDEX,
        query={"term": {"source": source}},
        refresh=True,
    )


def index_chunks(es: Elasticsearch, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
    actions = (
        {
            "_index": config.ES_INDEX,
            "_id": chunk.chunk_id,
            "_source": {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "embedding": embedding,
            },
        }
        for chunk, embedding in zip(chunks, embeddings)
    )
    success, _ = helpers.bulk(es, actions)
    es.indices.refresh(index=config.ES_INDEX)
    return success