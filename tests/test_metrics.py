import math

import pytest

from app.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank, relevance_flags

A = ("a.pdf", 1)
B = ("a.pdf", 2)
C = ("b.pdf", 5)


def test_a_relevant_page_only_counts_once():
    assert relevance_flags([A, A, C, B], {A, B}) == [1, 0, 0, 1]


def test_recall_at_k():
    flags = [1, 0, 0, 1]
    assert recall_at_k(flags, num_relevant=2, k=1) == 0.5
    assert recall_at_k(flags, num_relevant=2, k=4) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([0, 0, 1, 1], k=10) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0, 0], k=10) == 0.0
    assert reciprocal_rank([0, 0, 1], k=2) == 0.0  # the first hit is beyond the cutoff


def test_ndcg_rewards_higher_ranks():
    assert ndcg_at_k([1, 0, 0], num_relevant=1, k=10) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 0], num_relevant=1, k=10) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_zero_when_nothing_is_found():
    assert ndcg_at_k([0, 0, 0], num_relevant=1, k=10) == 0.0