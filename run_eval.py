"""CLI eval runner.

Usage:
    python run_eval.py                  # runs baseline
    python run_eval.py mmr              # runs one technique
    python run_eval.py --all            # runs all 5, prints comparison
"""
from __future__ import annotations

import argparse

from evaluation import config
from evaluation.vector_store import load_vectorstore
from evaluation.runner import TECHNIQUES, run_evaluation


def _print_summary(res: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  TECHNIQUE: {res['technique']}")
    print(f"{'='*70}")
    print(f"  Retrieval:")
    print(f"    MRR:        {res['retrieval']['MRR']:.3f}")
    print(f"    nDCG@5:     {res['retrieval']['nDCG@5']:.3f}")
    print(f"    Hit@5:      {res['retrieval']['Hit@5']:.0%}")
    print(f"  Answer (1-5):")
    print(f"    Accuracy:     {res['answer']['accuracy']:.2f}")
    print(f"    Completeness: {res['answer']['completeness']:.2f}")
    print(f"    Relevance:    {res['answer']['relevance']:.2f}")
    print(f"  Overall:      {res['overall_score']:.1%}")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")
    parser.add_argument("technique", nargs="?", default="baseline",
                        choices=TECHNIQUES, help="Which technique to evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Run every technique and print a comparison")
    args = parser.parse_args()

    config.validate()
    vs = load_vectorstore()

    if args.all:
        all_res = {}
        for t in TECHNIQUES:
            print(f"\n▶ Running [{t}]…")
            all_res[t] = run_evaluation(vs, t)
            _print_summary(all_res[t])
        # Final comparison
        print(f"{'='*70}")
        print(f"  COMPARISON (overall score)")
        print(f"{'='*70}")
        ranked = sorted(all_res.items(), key=lambda x: x[1]["overall_score"], reverse=True)
        for t, r in ranked:
            print(f"  {t:14s} {r['overall_score']:.1%}")
    else:
        res = run_evaluation(vs, args.technique)
        _print_summary(res)


if __name__ == "__main__":
    main()
