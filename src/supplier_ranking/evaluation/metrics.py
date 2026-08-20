"""Ranking metrics: NDCG@k, MAP@k, MRR, Precision@k, Recall@k."""

import numpy as np


def _order_by_score(relevance, scores):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(relevance)[order]


def dcg_at_k(relevance_sorted, k):
    relevance_sorted = np.asarray(relevance_sorted)[:k]
    if relevance_sorted.size == 0:
        return 0.0
    gains = 2 ** relevance_sorted - 1
    discounts = np.log2(np.arange(2, relevance_sorted.size + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevance, scores, k):
    rel_sorted = _order_by_score(relevance, scores)
    dcg = dcg_at_k(rel_sorted, k)
    ideal_sorted = np.sort(np.asarray(relevance))[::-1]
    idcg = dcg_at_k(ideal_sorted, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevance, scores, k, relevance_threshold):
    rel_sorted = _order_by_score(relevance, scores)[:k]
    if rel_sorted.size == 0:
        return 0.0
    return float(np.mean(rel_sorted >= relevance_threshold))


def recall_at_k(relevance, scores, k, relevance_threshold):
    relevance = np.asarray(relevance)
    total_relevant = np.sum(relevance >= relevance_threshold)
    if total_relevant == 0:
        return 0.0
    rel_sorted = _order_by_score(relevance, scores)[:k]
    hit = np.sum(rel_sorted >= relevance_threshold)
    return float(hit / total_relevant)


def average_precision_at_k(relevance, scores, k, relevance_threshold):
    rel_sorted = _order_by_score(relevance, scores)[:k]
    hits = 0
    sum_precisions = 0.0
    for i, rel in enumerate(rel_sorted, start=1):
        if rel >= relevance_threshold:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / hits if hits > 0 else 0.0


def reciprocal_rank(relevance, scores, relevance_threshold):
    rel_sorted = _order_by_score(relevance, scores)
    for i, rel in enumerate(rel_sorted, start=1):
        if rel >= relevance_threshold:
            return 1.0 / i
    return 0.0


def evaluate_ranking(groups, k=3, relevance_threshold=3):
    ndcgs, maps, mrrs, precisions, recalls = [], [], [], [], []
    for relevance, scores in groups:
        ndcgs.append(ndcg_at_k(relevance, scores, k))
        maps.append(average_precision_at_k(relevance, scores, k, relevance_threshold))
        mrrs.append(reciprocal_rank(relevance, scores, relevance_threshold))
        precisions.append(precision_at_k(relevance, scores, k, relevance_threshold))
        recalls.append(recall_at_k(relevance, scores, k, relevance_threshold))
    return {
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"MAP@{k}": float(np.mean(maps)),
        "MRR": float(np.mean(mrrs)),
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
    }
