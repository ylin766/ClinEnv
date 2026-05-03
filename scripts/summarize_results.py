"""Summarize evaluation results across all scored cases.

Reads eval/latest.json for each case+model and aggregates stage-level F1/R/P.

Usage:
    python3 scripts/summarize_results.py
    python3 scripts/summarize_results.py --model gpt-5.4
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

_CASES_DIR = Path("data/cases")


def summarize(models: list[str] | None = None):
    if models is None:
        # Auto-discover model slugs from evals/ subdirectories
        found: set[str] = set()
        for subject_dir in _CASES_DIR.iterdir():
            if not subject_dir.is_dir():
                continue
            for hadm_dir in subject_dir.iterdir():
                evals_dir = hadm_dir / "evals"
                if not evals_dir.is_dir():
                    continue
                for child in evals_dir.iterdir():
                    if child.is_dir() and (child / "latest.json").exists():
                        found.add(child.name)
        models = sorted(found)

    if not models:
        print("No evaluated cases found.")
        return

    stats: dict = defaultdict(lambda: {
        "f1_sum": 0.0, "recall_sum": 0.0, "precision_sum": 0.0,
        "count": 0, "stage_count": 0,
    })

    for subject_dir in _CASES_DIR.iterdir():
        if not subject_dir.is_dir():
            continue
        for hadm_dir in subject_dir.iterdir():
            if not hadm_dir.is_dir():
                continue
            for model_slug in models:
                eval_file = hadm_dir / "evals" / model_slug / "latest.json"
                if not eval_file.exists():
                    continue
                try:
                    data = json.loads(eval_file.read_text(encoding="utf-8"))
                    stage_scores = data.get("stages", [])
                    if not stage_scores:
                        continue
                    f1s  = [s.get("outcome_score", s).get("f1", 0.0) for s in stage_scores]
                    recs  = [s.get("outcome_score", s).get("recall", 0.0) for s in stage_scores]
                    precs = [s.get("outcome_score", s).get("precision", 0.0) for s in stage_scores]
                    stats[model_slug]["f1_sum"]        += sum(f1s) / len(f1s)
                    stats[model_slug]["recall_sum"]    += sum(recs) / len(recs)
                    stats[model_slug]["precision_sum"] += sum(precs) / len(precs)
                    stats[model_slug]["count"]         += 1
                    stats[model_slug]["stage_count"]   += len(stage_scores)
                except Exception:
                    continue

    print(f"\n{'Model':<20} | {'Cases':>6} | {'Stages':>7} | {'Avg F1':>8} | {'Recall':>8} | {'Precision':>9}")
    print("-" * 70)
    for model in models:
        s = stats[model]
        if s["count"] > 0:
            avg_f1   = s["f1_sum"]        / s["count"]
            avg_rec  = s["recall_sum"]    / s["count"]
            avg_prec = s["precision_sum"] / s["count"]
            print(f"{model:<20} | {s['count']:>6} | {s['stage_count']:>7} | {avg_f1:>8.3f} | {avg_rec:>8.3f} | {avg_prec:>9.3f}")
        else:
            print(f"{model:<20} | {'0':>6} | {'0':>7} | {'N/A':>8} | {'N/A':>8} | {'N/A':>9}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", help="Model slug(s) to summarize (default: auto-discover)")
    args = parser.parse_args()
    summarize(args.model)
