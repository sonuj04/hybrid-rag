"""Build evaluation/datasets/questions.json from candidates.json and your keep list.

Usage (from the project root):
    python -m scripts.build_dataset

It reads evaluation/datasets/keep.txt, one question per line:

    q004                                     a draft, unchanged
    q009 | your rewritten question           a draft with your own wording
    q012 | question | a.pdf:3; b.pdf:1       a draft with the correct pages replaced
    h01 | your own question | a.pdf:3        a hand-written question (page numbers: file.pdf:page)

Blank lines and lines starting with # are ignored.
"""
import json
import sys

from app import config
from app.es_client import get_es_client

DATASETS = config.PROJECT_ROOT / "evaluation" / "datasets"
CANDIDATES_PATH = DATASETS / "candidates.json"
KEEP_PATH = DATASETS / "keep.txt"
OUTPUT_PATH = DATASETS / "questions.json"


def parse_pages(text: str) -> list[dict]:
    pages = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        source, _, page = part.rpartition(":")
        if not source or not page.strip().isdigit():
            raise ValueError(f"bad page reference '{part}' (use file.pdf:3)")
        pages.append({"source": source.strip(), "page": int(page)})
    return pages


def page_exists(es, source: str, page: int) -> bool:
    result = es.count(
        index=config.ES_INDEX,
        query={"bool": {"filter": [{"term": {"source": source}}, {"term": {"page": page}}]}},
    )
    return result["count"] > 0


def main() -> None:
    if not KEEP_PATH.exists():
        print(f"{KEEP_PATH.relative_to(config.PROJECT_ROOT)} does not exist yet.")
        print("Create it with: python -m scripts.review_candidates --template > evaluation/datasets/keep.txt")
        sys.exit(1)

    candidates = {
        c["id"]: c for c in json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    }
    es = get_es_client()

    questions: list[dict] = []
    seen: set[str] = set()
    problems: list[str] = []

    for number, raw in enumerate(KEEP_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split("|")]
        question_id = fields[0]
        try:
            if question_id in seen:
                raise ValueError(f"'{question_id}' appears twice")
            if question_id in candidates:
                draft = candidates[question_id]
                question = fields[1] if len(fields) > 1 and fields[1] else draft["question"]
                relevant = parse_pages(fields[2]) if len(fields) > 2 and fields[2] else draft["relevant"]
            elif len(fields) >= 3 and fields[1] and fields[2]:
                question, relevant = fields[1], parse_pages(fields[2])
            else:
                raise ValueError(
                    f"'{question_id}' is not a draft id, so it needs: id | question | file.pdf:page"
                )
            for page in relevant:
                if not page_exists(es, page["source"], page["page"]):
                    raise ValueError(
                        f"no chunks found for {page['source']} page {page['page']} "
                        "(check the file name and the page number)"
                    )
        except ValueError as exc:
            problems.append(f"keep.txt line {number}: {exc}")
            continue
        seen.add(question_id)
        questions.append({"id": question_id, "question": question, "relevant": relevant})

    if problems:
        print("Fix these problems in keep.txt, then run again:\n")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    if not questions:
        print("keep.txt contains no questions.")
        sys.exit(1)

    OUTPUT_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    drafts = sum(1 for q in questions if q["id"] in candidates)
    print(
        f"Wrote {len(questions)} questions ({drafts} from drafts, {len(questions) - drafts} hand-written) "
        f"to {OUTPUT_PATH.relative_to(config.PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()