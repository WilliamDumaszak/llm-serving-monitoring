"""RAG evaluation metrics — hit_rate and MRR."""


def hit_rate(relevance_total: list[list[bool]]) -> float:
    if not relevance_total:
        return 0.0
    return sum(1 for line in relevance_total if True in line) / len(relevance_total)


def mrr(relevance_total: list[list[bool]]) -> float:
    if not relevance_total:
        return 0.0
    score = 0.0
    for line in relevance_total:
        for rank, relevant in enumerate(line):
            if relevant:
                score += 1 / (rank + 1)
                break
    return score / len(relevance_total)


def evaluate_search(search_fn, ground_truth: list[dict]) -> dict:
    """
    Evaluate a search function against ground truth.

    ground_truth: [{"query": "...", "doc_id": "..."}, ...]
    search_fn:    callable(query) -> list of dicts with 'doc_id'
    """
    relevance_total = []
    for item in ground_truth:
        results = search_fn(item["query"])
        relevance = [r.get("doc_id") == item["doc_id"] for r in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
        "n_queries": len(ground_truth),
    }
