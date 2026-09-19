"""Turns text into vectors using the sentence-transformers model from config."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app import config

# bge-small-en-v1.5 works best when SHORT QUERIES (not passages) start with this instruction.
# If you switch to a different embedding model, check its model card for the right prefixes.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    vector = get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    return vector.tolist()