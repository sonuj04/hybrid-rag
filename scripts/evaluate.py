"""Score the retrieval modes on a labelled question set.

Usage (from the project root):
    python -m scripts.evaluate
    python -m scripts.evaluate --typos --label typos
    python -m scripts.evaluate --modes dense hybrid --show-misses
"""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

from app import config
from app.evaluation.metrics import (
    ndcg_at_k,
    page_key,
    recall_at_k,
    reciprocal_rank,
    relevance_flags,
)
from app.evaluation.typos import add_typos
from app.generation.prompt import format_location
from app.retrieval.retriever import MODES, search

DEFAULT_DATASET = config.PROJECT_ROOT / "evaluation" / "datasets" / "questions.json"
RESULTS_DIR = config.PROJECT_ROOT / "evaluation" / "results"
MAX_K = 10
METRICS = ["recall@5", "recall@10", "mrr@10", "ndcg@10"]


def load_questions(path: Path) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    return [q for q in questions if q.get("relevant")]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def evaluate_mode(mode: str, questions: list[dict], use_typos: bool, show_misses: bool):
    search("warm up", mode=mode, k=1)  # load models once so the latency numbers are fair
    rows = []
    latencies = []
    for index, item in enumerate(questions):
        query = add_typos(item["question"], seed=index) if use_typos else item["question"]

        start = time.perf_counter()
        chunks = search(query, mode=mode, k=MAX_K)
        latencies.append((time.perf_counter() - start) * 1000)

        relevant = {(r["source"], r.get("page")) for r in item["relevant"]}
        flags = relevance_flags([page_key(chunk) for chunk in chunks], relevant)
        row = {
            "id": item["id"],
            "query": query,
            "recall@5": recall_at_k(flags, len(relevant), 5),
            "recall@10": recall_at_k(flags, len(relevant), 10),
            "mrr@10": reciprocal_rank(flags, 10),
            "ndcg@10": ndcg_at_k(flags, len(relevant), 10),
        }
        rows.append(row)

        if show_misses and row["recall@5"] == 0:
            print(f"\n[{mode}] MISS {item['id']}: {query}")
            print(f"  expected: {sorted(relevant, key=str)}")
            for rank, chunk in enumerate(chunks[:3], start=1):
                print(f"  got {rank}: {format_location(chunk)} | {chunk['text'][:100]}...")

    summary = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
    summary["latency_ms_mean"] = sum(latencies) / len(latencies)
    summary["latency_ms_p95"] = percentile(latencies, 0.95)
    return summary, rows


def print_table(results: dict[str, dict]) -> None:
    header = (
        f"{'mode':<8}{'recall@5':>10}{'recall@10':>11}{'mrr@10':>9}{'ndcg@10':>9}"
        f"{'latency ms mean/p95':>24}"
    )
    print("\n" + header)
    print("-" * len(header))
    for mode, s in results.items():
        latency = f"{s['latency_ms_mean']:.0f} / {s['latency_ms_p95']:.0f}"
        print(
            f"{mode:<8}{s['recall@5']:>10.3f}{s['recall@10']:>11.3f}"
            f"{s['mrr@10']:>9.3f}{s['ndcg@10']:>9.3f}{latency:>24}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval modes")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--typos", action="store_true", help="add typos to every question")
    parser.add_argument("--label", default="run", help="name used in the results file")
    parser.add_argument(
        "--show-misses", action="store_true", help="print questions whose page is not in the top 5"
    )
    parser.add_argument(
        "--id-prefix",
        default="",
        help="only use questions whose id starts with this (q = drafted, h = hand-written)",
    )
    args = parser.parse_args()

    questions = [q for q in load_questions(args.dataset) if q["id"].startswith(args.id_prefix)]
    if not questions:
        print(f"No usable questions in {args.dataset}. Did you create questions.json?")
        return
    suffix = " (with typos)" if args.typos else ""
    print(f"Evaluating {len(questions)} questions{suffix}")

    results: dict[str, dict] = {}
    per_question: dict[str, list] = {}
    for mode in args.modes:
        print(f"Running {mode} ...")
        results[mode], per_question[mode] = evaluate_mode(
            mode, questions, args.typos, args.show_misses
        )
    print_table(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}_{args.label}.json"
    payload = {
        "label": args.label,
        "timestamp": stamp,
        "num_questions": len(questions),
        "typos": args.typos,
        "config": {
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "hybrid_candidates": config.HYBRID_CANDIDATES,
        },
        "results": results,
        "per_question": per_question,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()