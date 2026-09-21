import math

from app.retrieval.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    retrieved = ["paper_a", "paper_x", "paper_b"]
    relevant = ["paper_a", "paper_b"]

    assert precision_at_k(retrieved, relevant, 2) == 0.5


def test_recall_at_k():
    retrieved = ["paper_a", "paper_x", "paper_b"]
    relevant = ["paper_a", "paper_b", "paper_c"]

    assert recall_at_k(retrieved, relevant, 3) == 2 / 3


def test_mrr():
    retrieved = ["paper_x", "paper_b", "paper_a"]
    relevant = ["paper_a", "paper_b"]

    assert mrr(retrieved, relevant) == 0.5


def test_ndcg_at_k():
    retrieved = ["paper_a", "paper_x", "paper_b"]
    relevant = ["paper_a", "paper_b"]

    actual_dcg = 1 + (1 / math.log2(4))
    ideal_dcg = 1 + (1 / math.log2(3))

    assert ndcg_at_k(retrieved, relevant, 3) == actual_dcg / ideal_dcg
