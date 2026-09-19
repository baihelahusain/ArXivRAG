def precision_at_k(retrieved, relevant, k):
    """
    Precision@K:
    Percentage of the top K retrieved papers that are relevant.
    """

    retrieved_at_k = retrieved[:k]

    if not retrieved_at_k:
        return 0.0

    relevant_count = 0

    for paper_id in retrieved_at_k:
        if paper_id in relevant:
            relevant_count += 1

    return relevant_count / len(retrieved_at_k)


def recall_at_k(retrieved, relevant, k):
    """
    Recall@K:
    Percentage of all relevant papers that were found
    in the top K results.
    """

    if not relevant:
        return 0.0

    retrieved_at_k = retrieved[:k]

    relevant_count = 0

    for paper_id in retrieved_at_k:
        if paper_id in relevant:
            relevant_count += 1

    return relevant_count / len(relevant)


def mrr(retrieved, relevant):
    """
    Mean Reciprocal Rank for a single query.

    Returns:
        1 / rank of the first relevant paper.
    """

    for rank, paper_id in enumerate(retrieved, start=1):
        if paper_id in relevant:
            return 1 / rank

    return 0.0


def dcg_at_k(retrieved, relevant, k):
    """
    Discounted Cumulative Gain.

    Relevant paper = 1
    Non-relevant paper = 0
    """

    score = 0.0

    for rank, paper_id in enumerate(retrieved[:k], start=1):

        if paper_id in relevant:
            score += 1 / __import__("math").log2(rank + 1)

    return score


def ndcg_at_k(retrieved, relevant, k):
    """
    Normalized Discounted Cumulative Gain.

    Measures whether relevant papers appear near the top
    of the ranking.
    """

    if not relevant:
        return 0.0

    actual_dcg = dcg_at_k(retrieved, relevant, k)

    ideal_results = list(relevant)
    ideal_dcg = dcg_at_k(ideal_results, relevant, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg