"""
Evaluator untuk Cosine Similarity.
Metrik: Precision@K, Recall@K, F1@K, MAP
"""
import numpy as np

def precision_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    top_k = ranked_ids[:k]
    if not top_k: return 0.0
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / len(top_k)

def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids: return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / len(relevant_ids)

def f1_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    p = precision_at_k(ranked_ids, relevant_ids, k)
    r = recall_at_k(ranked_ids, relevant_ids, k)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

def average_precision(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids: return 0.0
    top_k = ranked_ids[:k]
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(top_k, 1):
        if doc_id in relevant_ids:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(relevant_ids), k)

def evaluate_single_method(
    rankings: dict[str, list[int]], 
    ground_truth: dict[str, set[int]], 
    k: int = 10
) -> dict:
    precisions, recalls, f1s, aps = [], [], [], []

    for query, ranked in rankings.items():
        rel = ground_truth.get(query, set())
        precisions.append(precision_at_k(ranked, rel, k))
        recalls.append(recall_at_k(ranked, rel, k))
        f1s.append(f1_at_k(ranked, rel, k))
        aps.append(average_precision(ranked, rel, k))

    return {
        "precision@k": float(np.mean(precisions)),
        "recall@k":    float(np.mean(recalls)),
        "f1@k":        float(np.mean(f1s)),
        "map@k":       float(np.mean(aps)),
        "per_query": {
            "precision": [float(x) for x in precisions],
            "recall":    [float(x) for x in recalls],
            "f1":        [float(x) for x in f1s],
            "ap":        [float(x) for x in aps],
        },
    }