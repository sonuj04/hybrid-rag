"""Draft evaluation questions from random chunks of your documents.

Usage (from the project root):
    python -m scripts.generate_questions                 # up to 40 draft questions
    python -m scripts.generate_questions --n 60 --delay 6
    python -m scripts.generate_questions --fresh         # ignore earlier drafts, start over

Progress is saved after every question, so re-running simply continues where it stopped.
The output goes to evaluation/datasets/candidates.json. REVIEW IT before using it:
LLM-drafted questions are often vague, and they tend to reuse the passage's own words,
which makes keyword search look better than it really is.
"""
import argparse
import json
import random
import re
import time
from collections import defaultdict

import openai
from elasticsearch import helpers

from app import config
from app.es_client import get_es_client
from app.generation.llm import generate, is_daily_quota_error

OUTPUT_PATH = config.PROJECT_ROOT / "evaluation" / "datasets" / "candidates.json"

SYSTEM_PROMPT = """You write evaluation questions for a document search system.

Given a passage, write ONE question that the passage answers well.
Rules:
1. Someone who has NOT seen the passage must understand the question. Never say "this passage", "the text", "the paper", "the study", "the studies", "the research" or "the authors".
2. Ask about a specific fact, mechanism, measurement or relationship that the passage states. Include at least one concrete detail (a named method, condition, brain region, molecule, population or number), so that only this passage, and not a general overview of the topic, could answer it. Never ask a generic question such as "What is neuroplasticity?".
3. Paraphrase. Do not copy long phrases from the passage.
4. One sentence. Return only the question, nothing else."""


def is_good_passage(text: str) -> bool:
    """Skip reference lists, contact details and boilerplate."""
    if len(text) < 500:
        return False
    if re.search(r"http|\bdoi\b", text.lower()):
        return False
    digits = sum(character.isdigit() for character in text)
    return digits / len(text) < 0.06


def fetch_all_chunks(es) -> list[dict]:
    query = {"query": {"match_all": {}}, "_source": ["source", "page", "chunk_index", "text"]}
    return [hit["_source"] for hit in helpers.scan(es, query=query, index=config.ES_INDEX)]


def pick_passages(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """Pick n usable chunks, taking turns between documents so every paper is covered."""
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        if is_good_passage(chunk["text"]):
            by_source[chunk["source"]].append(chunk)
    for group in by_source.values():
        rng.shuffle(group)
    sources = sorted(by_source)
    rng.shuffle(sources)

    picked: list[dict] = []
    while len(picked) < n and any(by_source.values()):
        for source in sources:
            if by_source[source] and len(picked) < n:
                picked.append(by_source[source].pop())
    return picked


def draft_question(passage: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passage:\n{passage}\n\nQuestion:"},
    ]
    return generate(messages, temperature=0.3).strip().strip('"').strip()


def load_existing() -> list[dict]:
    if OUTPUT_PATH.exists():
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    return []


def save(candidates: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft evaluation questions")
    parser.add_argument("--n", type=int, default=40, help="target number of drafts in total")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=4.0, help="seconds to wait between LLM calls")
    parser.add_argument("--fresh", action="store_true", help="ignore earlier drafts and start over")
    args = parser.parse_args()

    candidates = [] if args.fresh else load_existing()
    done = {
        (c["relevant"][0]["source"], c["relevant"][0]["page"], c["passage"]) for c in candidates
    }

    chunks = fetch_all_chunks(get_es_client())
    passages = pick_passages(chunks, args.n, args.seed)
    todo = [c for c in passages if (c["source"], c.get("page"), c["text"]) not in done]
    print(f"{len(candidates)} drafts already saved. Drafting {len(todo)} more (target {args.n}).")

    for index, chunk in enumerate(todo, start=1):
        try:
            question = draft_question(chunk["text"])
        except openai.RateLimitError as exc:
            if is_daily_quota_error(exc):
                print("\nDaily quota for this model is used up. Progress is saved.")
                print("Switch model, use Ollama, or re-run after the quota resets.")
                break
            print(f"[{index}/{len(todo)}] FAILED: {exc}")
            continue
        except Exception as exc:  # keep going if one call fails
            print(f"[{index}/{len(todo)}] FAILED: {exc}")
            continue

        candidates.append(
            {
                "id": f"q{len(candidates) + 1:03d}",
                "question": question,
                "relevant": [{"source": chunk["source"], "page": chunk.get("page")}],
                "passage": chunk["text"],
            }
        )
        save(candidates)
        print(f"[{index}/{len(todo)}] {question}")
        time.sleep(args.delay)

    print(f"\n{len(candidates)} drafts are in {OUTPUT_PATH.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()