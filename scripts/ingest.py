"""Ingest documents from data/raw into Elasticsearch.

Usage (from the project root):
    python -m scripts.ingest              # ingest / re-ingest everything in data/raw
    python -m scripts.ingest --recreate   # delete the index first, then ingest
"""
import argparse

from app import config
from app.embeddings import embed_passages
from app.es_client import get_es_client
from app.ingestion.chunker import chunk_pages
from app.ingestion.indexer import (
    create_index_if_missing,
    delete_index,
    delete_source,
    index_chunks,
)
from app.ingestion.parser import SUPPORTED_EXTENSIONS, parse_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Elasticsearch")
    parser.add_argument("--recreate", action="store_true", help="delete the index before ingesting")
    args = parser.parse_args()

    files = sorted(
        p
        for p in config.RAW_DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"No .pdf/.md/.txt files found in {config.RAW_DATA_DIR}. Add some and re-run.")
        return

    es = get_es_client()
    if args.recreate:
        delete_index(es)
        print(f"Deleted index {config.ES_INDEX}")
    create_index_if_missing(es)

    total_chunks = 0
    for path in files:
        source = path.relative_to(config.RAW_DATA_DIR).as_posix()
        pages = parse_file(path, source)
        if not pages:
            print(f"[SKIP] {source}: no extractable text (scanned PDF?)")
            continue
        chunks = chunk_pages(pages)
        embeddings = embed_passages([chunk.text for chunk in chunks])
        delete_source(es, source)  # remove old chunks of this file so re-runs never duplicate
        index_chunks(es, chunks, embeddings)
        total_chunks += len(chunks)
        print(f"[OK] {source}: {len(pages)} pages -> {len(chunks)} chunks")

    count = es.count(index=config.ES_INDEX)["count"]
    print(f"\nDone. Ingested {total_chunks} chunks this run. Index '{config.ES_INDEX}' holds {count} chunks.")


if __name__ == "__main__":
    main()