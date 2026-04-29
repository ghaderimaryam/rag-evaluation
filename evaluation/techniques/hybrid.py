"""Technique 4 — Hybrid (BM25 + Dense): combines keyword + semantic retrieval."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever


NAME = "hybrid"
DESCRIPTION = (
    "Hybrid BM25 + dense — combines keyword search with semantic search via "
    "Reciprocal Rank Fusion. Big wins on queries with numbers, names, exact terms."
)


def get_retriever(vectorstore: Chroma, k: int = 5) -> BaseRetriever:
    # Lazy imports — surface clearer errors at call time.
    # langchain 1.0 moved EnsembleRetriever to langchain_classic; fall back for older installs.
    try:
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        from langchain.retrievers import EnsembleRetriever

    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    collection = vectorstore.get()
    docs_text = collection.get("documents") or []
    docs_meta = collection.get("metadatas") or []

    if not docs_text:
        raise RuntimeError(
            "Vector store returned no documents — can't build BM25 index. "
            "Is the chroma_db/ folder populated?"
        )

    documents = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(docs_text, docs_meta)
        if t and t.strip()
    ]

    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k

    dense = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])