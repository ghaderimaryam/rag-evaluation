"""Retrieval metrics: MRR, nDCG@k, Hit@k. No LLM calls — pure math."""
from __future__ import annotations

from typing import Optional

import numpy as np


def reciprocal_rank(retrieved: list[str], expected: set[str]) -> Optional[float]:
    """Reciprocal rank of the first relevant retrieved doc.

    Returns 1/k where k is the 1-indexed position of the first hit.
    Returns 0.0 if no hit. Returns None if the question has no expected
    answer (out-of-scope) — those should be excluded from the average.
    """
    if not expected:
        return None
    for i, source in enumerate(retrieved):
        if source in expected:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(relevance: list[float], k: int) -> float:
    """Discounted Cumulative Gain — rewards relevant docs near the top."""
    relevance = relevance[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))


def ndcg_at_k(retrieved: list[str], expected: set[str], k: int = 5) -> Optional[float]:
    """Normalized DCG@k — DCG divided by the ideal DCG. Range [0, 1].

    1.0 means all relevant docs were ranked perfectly at the top.
    0.0 means no relevant docs in top-k.
    None for out-of-scope questions.
    """
    if not expected:
        return None
    rel = [1.0 if s in expected else 0.0 for s in retrieved[:k]]
    ideal = sorted(rel, reverse=True)
    actual_dcg = dcg_at_k(rel, k)
    ideal_dcg = dcg_at_k(ideal, k)
    return 0.0 if ideal_dcg == 0 else actual_dcg / ideal_dcg


def hit_at_k(retrieved: list[str], expected: set[str], k: int = 5) -> Optional[bool]:
    """Did all expected docs appear in top-k? Binary.

    None for out-of-scope questions.
    """
    if not expected:
        return None
    return expected.issubset(set(retrieved[:k]))


def keyword_coverage(answer: str, keywords: list[str]) -> tuple[int, int]:
    """How many expected keywords appear in the answer (case-insensitive)?

    Returns (found, total).
    """
    if not keywords:
        return 0, 0
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return found, len(keywords)


def aggregate_retrieval(results: list[dict]) -> dict[str, float]:
    """Compute MRR, mean nDCG, and Hit Rate across all scorable queries.

    Out-of-scope queries (where expected_sources is empty) are excluded.
    """
    rr = [r["RR"] for r in results if isinstance(r["RR"], (int, float))]
    nd = [r["nDCG@5"] for r in results if isinstance(r["nDCG@5"], (int, float))]
    hits = [r["hit@5"] for r in results if isinstance(r["hit@5"], bool)]

    return {
        "MRR": float(np.mean(rr)) if rr else 0.0,
        "nDCG@5": float(np.mean(nd)) if nd else 0.0,
        "Hit@5": float(np.mean(hits)) if hits else 0.0,
        "scorable_queries": len(rr),
    }
