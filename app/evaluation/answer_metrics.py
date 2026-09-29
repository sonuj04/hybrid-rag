"""Answer-level checks for Version 6: does the model cite validly, stay grounded in the
retrieved text, and refuse correctly when the context doesn't answer the question?

Unlike app/evaluation/metrics.py (which scores retrieval), everything here scores what the
LLM did with the retrieved chunks.
"""
import json
import re

from app.generation.llm import judge
from app.generation.prompt import NO_ANSWER_TEXT, format_location

CITATION_RE = re.compile(r"\[(\d+)\]")

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker. You will see a QUESTION, an ANSWER, \
and the numbered SOURCE passages the answer was allowed to use.

For each claim in the ANSWER, decide whether it is directly supported by the SOURCE passage(s) \
it cites. A claim is unsupported if it adds information not in the cited passage, cites the \
wrong passage, or overstates what the passage says.

Reply with ONLY a JSON object, no other text, no markdown fences:
{"grounded": true or false, "unsupported_claims": ["..."], "reason": "one short sentence"}

"grounded" is true only if every claim in the ANSWER is supported. If the ANSWER is the exact \
sentence "I don't have enough information in the provided documents.", that is always grounded."""


def extract_citations(answer: str) -> list[int]:
    """All citation numbers used in the answer, e.g. 'a [1][3] b [2]' -> [1, 3, 2]."""
    return [int(n) for n in CITATION_RE.findall(answer)]


def invalid_citations(answer: str, num_chunks: int) -> list[int]:
    """Citation numbers the answer used that are outside the retrieved context (1..num_chunks)."""
    return sorted({n for n in extract_citations(answer) if n < 1 or n > num_chunks})


def is_refusal(answer: str) -> bool:
    return answer.strip() == NO_ANSWER_TEXT


def build_judge_messages(question: str, answer: str, chunks: list[dict]) -> list[dict]:
    blocks = [f"[{i}] ({format_location(c)})\n{c['text']}" for i, c in enumerate(chunks, start=1)]
    user_message = (
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCE passages:\n" + "\n\n".join(blocks)
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def judge_groundedness(question: str, answer: str, chunks: list[dict]) -> dict:
    """Ask the judge model whether the answer's claims are supported by the cited chunks.

    Returns {"grounded": bool, "unsupported_claims": list[str], "reason": str}. On a parse
    failure, returns grounded=False with the raw text in "reason", so a bad response counts
    as a failure instead of silently passing.
    """
    if is_refusal(answer):
        return {"grounded": True, "unsupported_claims": [], "reason": "correct refusal"}
    raw = judge(build_judge_messages(question, answer, chunks))
    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(cleaned)
        return {
            "grounded": bool(result.get("grounded", False)),
            "unsupported_claims": list(result.get("unsupported_claims", [])),
            "reason": str(result.get("reason", "")),
        }
    except (json.JSONDecodeError, AttributeError):
        return {
            "grounded": False,
            "unsupported_claims": [],
            "reason": f"unparseable judge output: {raw[:200]}",
        }