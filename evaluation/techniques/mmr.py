"""Technique 2 — MMR (Maximal Marginal Relevance): diversifies retrieved chunks.

Problem it fixes: plain similarity search can return 5 chunks all from the SAME
vendor (because each chunk is highly similar to the query individually).
This wastes the top-5 slots.

How: MMR scores each candidate not just by relevance to the query, but ALSO
penalizes similarity to chunks already chosen. Result: top-5 covers more vendors.

Cost: same as baseline (no extra LLM/embedding calls).
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever


NAME = "mmr"
DESCRIPTION = (
    "Maximal Marginal Relevance — diversifies results by penalizing chunks similar "
    "to ones already selected. Same cost as baseline."
)


def get_retriever(vectorstore: Chroma, k: int = 5) -> BaseRetriever:
    return vectorstore.as_retriever(
        search_type="mmr",
        # fetch_k = how many candidates to consider before diversifying
        # lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity. 0.5 is a balanced default.
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5},
    )
