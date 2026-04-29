"""Technique 5 — LLM Reranker: retrieve top-15, ask LLM to score each, keep top-5.

Problem it fixes: similarity score is a rough proxy for relevance. Two chunks
might have similar embedding distances but very different actual usefulness for
the question. A language model reading both can tell which is better.

How: pull more candidates than we need (15), then ask a small LLM to score each
candidate's relevance to the question on 0-10. Sort by score, return top-k.

Cost: 1 extra LLM call (gpt-4o-mini scoring 15 chunks ≈ $0.0005). Slow-ish.
This is the most accurate technique and also the most expensive.
"""
from __future__ import annotations

import json
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from pydantic import Field


NAME = "reranker"
DESCRIPTION = (
    "LLM Reranker — retrieves top-15 candidates, asks gpt-4o-mini to score each "
    "for relevance, keeps the top-5. Most accurate, ~1 extra LLM call per query."
)


_RERANK_PROMPT = """Score how relevant each numbered passage is to the question, 0-10.

Question: {question}

Passages:
{passages}

Return ONLY a JSON array of integers, one per passage, in order. Example: [8, 2, 9, 0, 5]"""


class LLMRerankRetriever(BaseRetriever):
    """Retriever that pulls extra candidates and reranks them with an LLM."""

    base_retriever: BaseRetriever
    llm: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(model="gpt-4o-mini", temperature=0))
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # 1. Pull a larger candidate pool.
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []

        # 2. Ask the LLM to score each passage 0-10.
        passages = "\n\n".join(
            f"[{i}] {doc.page_content[:500]}"
            for i, doc in enumerate(candidates)
        )
        prompt = _RERANK_PROMPT.format(question=query, passages=passages)
        response = self.llm.invoke([HumanMessage(content=prompt)])

        # 3. Parse the JSON list of scores. Fall back to the original order on parse fail.
        try:
            text = response.content.strip()
            if text.startswith("```"):
                text = text.strip("`").lstrip("json").strip()
            scores = json.loads(text)
            if not isinstance(scores, list) or len(scores) != len(candidates):
                return candidates[: self.top_k]
        except (json.JSONDecodeError, AttributeError):
            return candidates[: self.top_k]

        # 4. Sort candidates by score descending, return top-k.
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]


def get_retriever(vectorstore: Chroma, k: int = 5) -> BaseRetriever:
    base = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 15}
    )
    return LLMRerankRetriever(base_retriever=base, top_k=k)
