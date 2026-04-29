"""Run the evaluation suite for a given retrieval technique."""
from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm

from evaluation import metrics
from evaluation.judge import judge_answer
from evaluation.test_cases import ALL_CASES

# ─── Available techniques registry ─────────────────────────────────────────
TECHNIQUES = ["baseline", "mmr", "multi_query", "hybrid", "reranker"]


def get_technique(name: str):
    """Dynamically import a technique module by name."""
    if name not in TECHNIQUES:
        raise ValueError(f"Unknown technique: {name}. Choose from {TECHNIQUES}")
    return importlib.import_module(f"evaluation.techniques.{name}")


# ─── RAG chain (same prompt as the chat UI uses) ───────────────────────────
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful vendor recommendation assistant for event planning.
Use the following retrieved context from our vendor knowledge base to answer the user's question.
If you don't know the answer or the context doesn't contain relevant information, say so honestly.
Always mention specific vendor names, prices, ratings, and other details when available.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ─── Per-question evaluation ───────────────────────────────────────────────
def _evaluate_one(tc: dict, retriever, llm, chain) -> dict:
    """Run one test case and compute all metrics. Returns a flat dict."""
    q = tc["question"]
    expected = set(tc["expected_sources"])
    difficulty = tc["difficulty"]

    # 1. Retrieval
    docs = retriever.invoke(q)
    retrieved_sources = [doc.metadata.get("source", "?") for doc in docs]
    context = _format_docs(docs)

    # 2. Retrieval metrics
    rr = metrics.reciprocal_rank(retrieved_sources, expected)
    nd = metrics.ndcg_at_k(retrieved_sources, expected, k=5)
    hit = metrics.hit_at_k(retrieved_sources, expected, k=5)

    # 3. Generation
    answer = chain.invoke({"context": context, "chat_history": [], "question": q})

    # 4. Answer metrics
    kw_found, kw_total = metrics.keyword_coverage(answer, tc["expected_keywords"])
    judge_scores = judge_answer(q, answer, context, difficulty)

    return {
        "question": q,
        "difficulty": difficulty,
        "expected_sources": ", ".join(expected) if expected else "(out-of-scope)",
        "retrieved_top5": ", ".join(retrieved_sources[:5]),
        "RR": round(rr, 3) if rr is not None else None,
        "nDCG@5": round(nd, 3) if nd is not None else None,
        "hit@5": hit,
        "accuracy": judge_scores.get("accuracy", 0),
        "completeness": judge_scores.get("completeness", 0),
        "relevance": judge_scores.get("relevance", 0),
        "keyword_coverage": f"{kw_found}/{kw_total}",
        "answer_preview": answer[:200],
        "judge_reasoning": judge_scores.get("reasoning", ""),
    }


# ─── Main entry point ──────────────────────────────────────────────────────
def run_evaluation(
    vectorstore: Chroma,
    technique_name: str,
    test_cases: Optional[list[dict]] = None,
    max_workers: int = 4,
    progress: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Run the full eval suite for one technique.

    Args:
        vectorstore: loaded Chroma instance
        technique_name: one of TECHNIQUES
        test_cases: defaults to ALL_CASES
        max_workers: parallel question evaluation (each makes 2-3 LLM calls)
        progress: show tqdm bar in terminal
        progress_callback: called as (done, total) after each test — use for UI updates

    Returns:
        {
            "technique": str,
            "per_question": list[dict],
            "retrieval": {"MRR": float, "nDCG@5": float, "Hit@5": float, ...},
            "answer": {"accuracy": float, "completeness": float, "relevance": float, "overall": float},
            "by_difficulty": {"easy": {...}, "medium": {...}, "hard": {...}},
        }
    """
    test_cases = test_cases or ALL_CASES

    # Build the retriever and chain for this technique
    technique = get_technique(technique_name)
    retriever = technique.get_retriever(vectorstore, k=5)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    chain = _RAG_PROMPT | llm | StrOutputParser()

    # Run questions in parallel — but not too parallel (we'd hit rate limits)
    results: list[Optional[dict]] = [None] * len(test_cases)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_evaluate_one, tc, retriever, llm, chain): i
            for i, tc in enumerate(test_cases)
        }
        iterator = as_completed(futures)
        if progress:
            iterator = tqdm(iterator, total=len(test_cases), desc=f"Evaluating [{technique_name}]")
        for fut in iterator:
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"⚠️  Q{i} failed: {e}")
                results[i] = {"question": test_cases[i]["question"], "error": str(e)}
            done += 1
            if progress_callback:
                progress_callback(done, len(test_cases))

    # Filter out errored rows for aggregation
    clean = [r for r in results if r and "error" not in r]

    # Aggregate
    retrieval_summary = metrics.aggregate_retrieval(clean)

    import numpy as np
    acc = np.mean([r["accuracy"] for r in clean]) if clean else 0
    comp = np.mean([r["completeness"] for r in clean]) if clean else 0
    rel = np.mean([r["relevance"] for r in clean]) if clean else 0

    by_diff = {}
    for d in ["easy", "medium", "hard"]:
        subset = [r for r in clean if r["difficulty"] == d]
        if subset:
            by_diff[d] = {
                "accuracy": float(np.mean([r["accuracy"] for r in subset])),
                "completeness": float(np.mean([r["completeness"] for r in subset])),
                "relevance": float(np.mean([r["relevance"] for r in subset])),
                "n": len(subset),
            }

    overall_score = (
        retrieval_summary["Hit@5"] * 0.2
        + retrieval_summary["MRR"] * 0.1
        + retrieval_summary["nDCG@5"] * 0.1
        + (acc / 5) * 0.2
        + (comp / 5) * 0.2
        + (rel / 5) * 0.2
    )

    return {
        "technique": technique_name,
        "description": technique.DESCRIPTION,
        "per_question": clean,
        "retrieval": retrieval_summary,
        "answer": {
            "accuracy": float(acc),
            "completeness": float(comp),
            "relevance": float(rel),
        },
        "by_difficulty": by_diff,
        "overall_score": float(overall_score),
    }
