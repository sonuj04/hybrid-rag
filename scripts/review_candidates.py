"""Helps you review the drafted questions.

Usage (from the project root):
    python -m scripts.review_candidates > evaluation/review.txt
    python -m scripts.review_candidates --template > evaluation/datasets/keep.txt

The first command writes each question next to the passage it came from, so you can judge it.
The second writes your editable keep list. Run the second command only ONCE:
running it again overwrites your edits.
"""
import argparse
import json
import re
import textwrap

from app import config

CANDIDATES_PATH = config.PROJECT_ROOT / "evaluation" / "datasets" / "candidates.json"
SUSPICIOUS = re.compile(
    r"\b(the|this) (study|studies|research|paper|authors?|text|passage|article)\b"
    r"|\bthe (two|three|four|five) \w+",
    re.IGNORECASE,
)


def location(item: dict) -> str:
    relevant = item["relevant"][0]
    if relevant.get("page"):
        return f"{relevant['source']}, page {relevant['page']}"
    return relevant["source"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Review drafted questions")
    parser.add_argument("--template", action="store_true", help="print the editable keep list")
    args = parser.parse_args()

    candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))

    if args.template:
        print("# One question per line. DELETE the lines you do not want. Edit question text in place.")
        print("# Add your own questions at the bottom, like:  h01 | your question | file.pdf:3")
        print()
        for item in candidates:
            print(f"{item['id']} | {item['question'].replace('|', '/')}")
        return

    for item in candidates:
        flag = "   [!] check the wording" if SUSPICIOUS.search(item["question"]) else ""
        print(f"{item['id']}  {item['question']}{flag}")
        print(f"      from: {location(item)}")
        print(
            textwrap.fill(
                item["passage"][:400],
                width=100,
                initial_indent="      passage: ",
                subsequent_indent="               ",
            )
            + " ..."
        )
        print()


if __name__ == "__main__":
    main()