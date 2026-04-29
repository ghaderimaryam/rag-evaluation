"""Open the persisted Chroma index. The index must already exist on disk."""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from evaluation import config


def load_vectorstore(chroma_path: Path | None = None) -> Chroma:
    path = chroma_path or config.CHROMA_PATH
    return Chroma(
        persist_directory=str(path),
        embedding_function=OpenAIEmbeddings(model=config.MODEL_EMBED),
        collection_name=config.COLLECTION_NAME,
    )
