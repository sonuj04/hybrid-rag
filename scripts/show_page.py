"""Print all chunks of one page.

Usage (from the project root):
    python -m scripts.show_page neuroplasticity_13.pdf 1
"""
import argparse

from app import config
from app.es_client import get_es_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Show all chunks of one page")
    parser.add_argument("source", help="file name, for example neuroplasticity_13.pdf")
    parser.add_argument("page", type=int, help="page number, for example 1")
    args = parser.parse_args()

    response = get_es_client().search(
        index=config.ES_INDEX,
        query={
            "bool": {
                "filter": [{"term": {"source": args.source}}, {"term": {"page": args.page}}]
            }
        },
        sort=[{"chunk_index": "asc"}],
        source=["chunk_index", "text"],
        size=50,
    )
    hits = response["hits"]["hits"]
    if not hits:
        print(f"No chunks found for {args.source} page {args.page}.")
        return
    for hit in hits:
        print(f"--- chunk {hit['_source']['chunk_index']} ---")
        print(hit["_source"]["text"])
        print()


if __name__ == "__main__":
    main()