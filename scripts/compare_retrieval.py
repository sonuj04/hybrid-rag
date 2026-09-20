"""Show what each retrieval mode returns for the same question.

Usage (from the project root):
    python -m scripts.compare_retrieval "your question here"
    python -m scripts.compare_retrieval "your question here" --k 8
"""
import argparse

from app import config
from app.generation.prompt import format_location
from app.retrieval.retriever import MODES, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense, BM25 and hybrid retrieval")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    for mode in MODES:
        print(f"\n=== {mode.upper()} ===")
        for rank, chunk in enumerate(search(args.question, mode=mode, k=args.k), start=1):
            snippet = chunk["text"][:90].replace("\n", " ")
            print(f"{rank}. score={chunk['score']:.4f}  {format_location(chunk)}  |  {snippet}...")


if __name__ == "__main__":
    main()