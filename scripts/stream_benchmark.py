"""Stream benchmark runner.

Continuously scans data/cases/ for cases that have a plan but are missing
model output, and runs them in parallel. Reads target model from config/runtime.json.

Usage:
    python3 scripts/stream_benchmark.py
    python3 scripts/stream_benchmark.py --models gpt-5.4 gpt-5.4-mini
    python3 scripts/stream_benchmark.py --workers 4
"""

import argparse
import json
import os
import time
import subprocess
import concurrent.futures
from pathlib import Path

_CASES_DIR   = Path("data/cases")
_CONFIG_FILE = Path("config/runtime.json")


def _load_models(override: list[str] | None = None) -> list[str]:
    if override:
        return override
    try:
        conf = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        model = conf.get("mut_model")
        return [model] if model else ["gpt-5.4"]
    except Exception:
        return ["gpt-5.4"]


def get_ready_cases(models: list[str]) -> list[tuple[str, str, str]]:
    """Find cases that have a case.json but are missing output for at least one model."""
    ready_to_run = []
    for subject_dir in _CASES_DIR.iterdir():
        if not subject_dir.is_dir():
            continue
        for hadm_dir in subject_dir.iterdir():
            if not (hadm_dir / "case" / "case.json").exists():
                continue
            for model in models:
                model_slug = model.replace("/", "-").replace(":", "-")
                if not (hadm_dir / "runs" / model_slug / "output.json").exists():
                    ready_to_run.append((subject_dir.name, hadm_dir.name, model))
    return ready_to_run

def run_single_case(subject_id: str, hadm_id: str, model: str) -> bool:
    """Run one benchmark case."""
    print(f"[STREAM] Starting: {model} | {subject_id}/{hadm_id}", flush=True)
    cmd = [
        "python3", "-m", "env.main",
        "--subject-id", subject_id,
        "--hadm-id",    hadm_id,
        "--model",      model,
        "--quiet",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[STREAM] Done: {model} | {subject_id}/{hadm_id}", flush=True)
        return True
    except Exception as e:
        print(f"[STREAM] Failed: {model} | {subject_id}/{hadm_id} -> {e}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models",  nargs="+", help="Model(s) to run (default: from config/runtime.json)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    args = parser.parse_args()

    models       = _load_models(args.models)
    max_workers  = args.workers
    failed_tasks: set = set()

    print(f"Stream Benchmark started. models={models} workers={max_workers}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        active: dict = {}

        while True:
            # Clean up finished futures
            for f in [f for f in active if f.done()]:
                if not f.result():
                    failed_tasks.add(active[f])
                del active[f]

            # Find and submit new work
            for sid, hid, model in get_ready_cases(models):
                if len(active) >= max_workers:
                    break
                task_id = (sid, hid, model)
                if task_id in failed_tasks:
                    continue
                if any(v == task_id for v in active.values()):
                    continue
                active[executor.submit(run_single_case, sid, hid, model)] = task_id

            if not get_ready_cases(models) and not active:
                print("No pending cases. Sleeping 10s...", flush=True)
                time.sleep(10)
            else:
                time.sleep(2)


if __name__ == "__main__":
    main()
