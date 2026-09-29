"""Version 6: evaluate the LLM's answers, not just retrieval.

Checks three things per mode="rerank" answer:
  1. citation validity  - every [n] in the answer points at a chunk that was retrieved
  2. groundedness        - the judge model checks whether cited claims are supported
  3. refusal correctness - on questions the corpus can't answer, does it refuse verbatim

Uses Gemini (LLM_*) to answer and local Ollama (JUDGE_*) to grade, so grading is free.
Each answerable and unanswerable question spends one Gemini call.

Usage (from the project root):
    python -m scripts.evaluate_answers
    python -m scripts.evaluate_answers --limit-answerable 6 --limit-unanswerable 4
"""
import argparse
import json
from datetime import datetime
import openai

from app import config
from app.evaluation.answer_metrics import invalid_citations, is_refusal, judge_groundedness
from app.generation.llm import generate, is_daily_quota_error
from app.generation.prompt import build_messages
from app.retrieval.retriever import search

QUESTIONS_PATH = config.PROJECT_ROOT / "evaluation" / "datasets" / "questions.json"
UNANSWERABLE_PATH = config.PROJECT_ROOT / "evaluation" / "datasets" / "unanswerable.json"
RESULTS_DIR = config.PROJECT_ROOT / "evaluation" / "results"
MODE = "rerank"
K = 5


def load_answerable(limit: int) -> list[dict]:
    """Hand-written questions only ('h' ids): their ground-truth pages were checked by hand,
    so a judge disagreement is more likely to be a real problem than a labelling error."""
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    handwritten = [q for q in questions if q["id"].startswith("h") and q.get("relevant")]
    return handwritten[:limit]


def load_unanswerable(limit: int) -> list[dict]:
    if not UNANSWERABLE_PATH.exists():
        return []
    return json.loads(UNANSWERABLE_PATH.read_text(encoding="utf-8"))[:limit]


def run_answerable(questions: list[dict]) -> tuple[list[dict], str | None]:
    """Returns (rows, stop_reason). stop_reason is None if every question was answered,
    "quota" if Gemini's daily quota was hit, or "api_error" if Gemini kept failing after
    generate()'s own retries (e.g. a 503 'high demand' outage). Either way, rows holds only
    what completed before stopping, so a failure doesn't lose calls already spent."""
    rows = []
    for item in questions:
        chunks = search(item["question"], mode=MODE, k=K)
        try:
            answer = generate(build_messages(item["question"], chunks))
        except openai.RateLimitError as exc:
            if is_daily_quota_error(exc):
                print(f"[STOP] {item['id']}: daily Gemini quota reached; stopping here")
                return rows, "quota"
            print(f"[STOP] {item['id']}: rate limited after retries; stopping here")
            return rows, "api_error"
        except (openai.InternalServerError, openai.APIConnectionError) as exc:
            print(f"[STOP] {item['id']}: Gemini API error after retries "
                  f"({type(exc).__name__}); stopping here")
            return rows, "api_error"
        bad_citations = invalid_citations(answer, len(chunks))
        verdict = judge_groundedness(item["question"], answer, chunks)
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "citations_valid": bad_citations == [],
            "invalid_citations": bad_citations,
            "grounded": verdict["grounded"],
            "unsupported_claims": verdict["unsupported_claims"],
            "judge_reason": verdict["reason"],
        })
        status = "OK" if bad_citations == [] and verdict["grounded"] else "ISSUE"
        print(f"[{status}] {item['id']}: {item['question'][:70]}")
    return rows, None


def run_unanswerable(questions: list[dict]) -> tuple[list[dict], bool]:
    """Returns (rows, hit_daily_quota); see run_answerable for what hit_daily_quota means."""
    rows = []
    for item in questions:
        chunks = search(item["question"], mode=MODE, k=K)
        try:
            answer = generate(build_messages(item["question"], chunks))
        except openai.RateLimitError as exc:
            if is_daily_quota_error(exc):
                print(f"[STOP] {item['id']}: daily Gemini quota reached; stopping here")
                return rows, True
            raise
        refused = is_refusal(answer)
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "answer": answer,
            "refused": refused,
        })
        print(f"[{'OK' if refused else 'ISSUE'}] {item['id']}: {item['question'][:70]}")
    return rows, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM answer quality (Version 6)")
    parser.add_argument("--limit-answerable", type=int, default=8)
    parser.add_argument("--limit-unanswerable", type=int, default=6)
    args = parser.parse_args()

    answerable = load_answerable(args.limit_answerable)
    unanswerable = load_unanswerable(args.limit_unanswerable)
    if not answerable and not unanswerable:
        print("No questions to evaluate. Check evaluation/datasets/questions.json and unanswerable.json.")
        return

    print(f"Answering {len(answerable)} answerable questions (mode={MODE}, k={K}) ...")
    answerable_rows, stop_reason = run_answerable(answerable)
    if stop_reason:
        print(f"\nSkipping unanswerable questions: stopped early ({stop_reason})")
        unanswerable_rows = []
    else:
        print(f"\nAnswering {len(unanswerable)} unanswerable questions ...")
        unanswerable_rows, stop_reason = run_unanswerable(unanswerable)

    citation_rate = (
        sum(r["citations_valid"] for r in answerable_rows) / len(answerable_rows)
        if answerable_rows else None
    )
    groundedness_rate = (
        sum(r["grounded"] for r in answerable_rows) / len(answerable_rows)
        if answerable_rows else None
    )
    refusal_rate = (
        sum(r["refused"] for r in unanswerable_rows) / len(unanswerable_rows)
        if unanswerable_rows else None
    )

    print("\nSUMMARY")
    if citation_rate is not None:
        print(f"  citation validity:   {citation_rate:.2f}  ({len(answerable_rows)} questions)")
        print(f"  groundedness:        {groundedness_rate:.2f}  ({len(answerable_rows)} questions)")
    if refusal_rate is not None:
        print(f"  refusal correctness: {refusal_rate:.2f}  ({len(unanswerable_rows)} questions)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}_answer-eval.json"
    payload = {
        "timestamp": stamp,
        "mode": MODE,
        "k": K,
        "stop_reason": stop_reason,
        "summary": {
            "citation_validity_rate": citation_rate,
            "groundedness_rate": groundedness_rate,
            "refusal_rate": refusal_rate,
        },
        "answerable": answerable_rows,
        "unanswerable": unanswerable_rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if stop_reason == "quota":
        print("Hit the daily Gemini quota partway through. Re-run this script once it resets"
              " (check the AI Studio dashboard for your project's reset time).")
    elif stop_reason == "api_error":
        print("Gemini's API kept failing after retries — likely a temporary upstream outage,"
              " not your code or quota. Wait a while and re-run. If it persists for a long"
              " time, check https://discuss.ai.google.dev for known incidents.")


if __name__ == "__main__":
    main()