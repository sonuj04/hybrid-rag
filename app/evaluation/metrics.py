"""Retrieval metrics for page-level relevance judgments.

A "relevant page" is a (source file, page number) pair. A retrieved chunk counts as a hit
if it comes from a relevant page. If several chunks come from the same relevant page,
only the first one counts, so scores can never exceed 1.
"""
import math

Page = tuple[str, int | None]


def page_key(chunk: dict) -> Page:
    return (chunk["source"], chunk.get("page"))


def relevance_flags(retrieved: list[Page], relevant: set[Page]) -> list[int]:
    """1 the first time a relevant page shows up in the ranking, otherwise 0."""
    seen: set[Page] = set()
    flags = []
    for page in retrieved:
        if page in relevant and page not in seen:
            flags.append(1)
            seen.add(page)
        else:
            flags.append(0)
    return flags


def recall_at_k(flags: list[int], num_relevant: int, k: int) -> float:
    """Fraction of the relevant pages that appear in the top k."""
    return sum(flags[:k]) / num_relevant


def reciprocal_rank(flags: list[int], k: int) -> float:
    """1 / rank of the first hit within the top k, or 0 if there is none."""
    for rank, flag in enumerate(flags[:k], start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(flags: list[int], num_relevant: int, k: int) -> float:
    """Normalized discounted cumulative gain: rewards ranking relevant pages high."""
    dcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(flags[:k], start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(num_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0