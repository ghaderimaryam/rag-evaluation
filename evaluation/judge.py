"""LLM-as-judge: scores generated answers on Accuracy, Completeness, Relevance (1-5)."""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# temperature=0 → deterministic scoring across runs
_judge_llm: Optional[ChatOpenAI] = None


def _get_judge() -> ChatOpenAI:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _judge_llm


_PROMPT = """You are a strict evaluation judge for a RAG system that answers questions about event vendors.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}
Difficulty level: {difficulty}

Score the answer on three criteria (1-5 each). Be STRICT — do not default to high scores:

1. **Accuracy**: Are ALL facts correct and supported by the context? Deduct for any hallucinated details.
   5 = every fact verified in context, 3 = mostly correct with minor gaps, 1 = wrong or hallucinated

2. **Completeness**: Does the answer cover ALL relevant vendors/details from context?
   For comparison questions: did it mention ALL matching vendors? For out-of-scope: did it correctly refuse?
   5 = exhaustive, 3 = partial coverage, 1 = missing most info

3. **Relevance**: Does the answer directly address the specific question asked?
   5 = precisely answers what was asked, 3 = somewhat related, 1 = off-topic

IMPORTANT: For out-of-scope questions (weather, booking, info we can't have), a GOOD answer
should clearly state it cannot help — score Accuracy=5 and Relevance=5 if it refuses appropriately.
A BAD answer hallucinates an answer — score Accuracy=1.

Return ONLY a JSON object:
{{"accuracy": <int>, "completeness": <int>, "relevance": <int>, "reasoning": "one sentence"}}"""


def judge_answer(
    question: str, answer: str, context: str, difficulty: str
) -> dict:
    """Score one (question, answer) pair. Returns dict with int scores + reasoning."""
    prompt = _PROMPT.format(
        question=question,
        context=context[:4000],
        answer=answer,
        difficulty=difficulty,
    )
    response = _get_judge().invoke([HumanMessage(content=prompt)])
    try:
        # Models sometimes wrap JSON in markdown fences; strip them.
        text = response.content.strip()
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "reasoning": "Failed to parse judge response",
        }
