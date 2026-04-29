"""Project configuration. Loads from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vector store — copy this folder over from your RAG project
DATA_DIR    = Path(os.getenv("DATA_DIR",    str(PROJECT_ROOT / "data")))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma_db")))

# Models — must match what the index was built with
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL_GEN       = os.getenv("MODEL_GEN", "gpt-4o-mini")
MODEL_EMBED     = os.getenv("MODEL_EMBED", "text-embedding-3-small")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vendors")


def validate() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"No vector store at {CHROMA_PATH}.\n"
            f"Copy your `chroma_db/` folder from your RAG project into {DATA_DIR}/."
        )
