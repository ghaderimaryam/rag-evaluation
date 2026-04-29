"""Technique 1 — BASELINE: plain cosine-similarity search.

This is the technique your current chat UI uses. It's our control group:
every other technique is measured against this.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever


NAME = "baseline"
DESCRIPTION = (
    "Plain cosine-similarity search. Returns top-5 most semantically similar chunks. "
    "Fast and cheap, but struggles with numbers, exact names, and abstract queries."
)


def get_retriever(vectorstore: Chroma, k: int = 5) -> BaseRetriever:
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
