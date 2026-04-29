"""Technique 3 — Multi-Query: rephrases the question N ways, retrieves for each, merges."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever


NAME = "multi_query"
DESCRIPTION = (
    "Multi-Query — rephrases the question 3 ways, retrieves for each, merges results. "
    "Improves robustness to phrasing. Adds 1 LLM call."
)


def get_retriever(vectorstore: Chroma, k: int = 5) -> BaseRetriever:
    # langchain 1.0 moved MultiQueryRetriever to langchain_classic; fall back for older installs.
    try:
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    except ImportError:
        from langchain.retrievers.multi_query import MultiQueryRetriever

    from langchain_openai import ChatOpenAI

    base = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return MultiQueryRetriever.from_llm(retriever=base, llm=llm)