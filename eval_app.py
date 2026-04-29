"""Launch the RAG evaluation dashboard.

Usage:
    python eval_app.py
"""
from __future__ import annotations

from evaluation import config
from evaluation.vector_store import load_vectorstore
from evaluation.ui import build_demo


def main() -> None:
    config.validate()
    vs = load_vectorstore()
    demo, css, theme = build_demo(vs)
    demo.launch(theme=theme, css=css, inbrowser=True)


if __name__ == "__main__":
    main()
