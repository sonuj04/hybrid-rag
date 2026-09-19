"""Ask a question about your documents.

Usage (from the project root):
    python -m scripts.ask "your question here"
    python -m scripts.ask "your question here" --k 8 --show-context
"""
import argparse

from app import config
from app.generation.llm import generate
from app.generation.prompt import build_messages, format_location
from app.retrieval.dense import search_dense


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about your documents")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="number of chunks to retrieve")
    parser.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    args = parser.parse_args()

    chunks = search_dense(args.question, k=args.k)
    if not chunks:
        print("Nothing retrieved. Did you run: python -m scripts.ingest ?")
        return

    if args.show_context:
        print("\nRETRIEVED CONTEXT")
        for number, chunk in enumerate(chunks, start=1):
            print(f"\n[{number}] score={chunk['score']:.3f}  {format_location(chunk)}")
            print(chunk["text"])

    answer = generate(build_messages(args.question, chunks))

    print("\nANSWER")
    print(answer)
    print("\nSOURCES")
    for number, chunk in enumerate(chunks, start=1):
        print(f"[{number}] {format_location(chunk)}")


if __name__ == "__main__":
    main()