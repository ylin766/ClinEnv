"""Standalone evaluation entry point.

Reads output.json (env run result) and writes scored results to:
  data/cases/<subject_id>/<hadm_id>/evals/<model>/latest.json
  data/cases/<subject_id>/<hadm_id>/evals/<model>/<YYYYMMDD_HHMMSS>.json

output.json is never modified.

Usage:
    python -m evaluation.main --subject-id 10196757 --hadm-id 24725711 --model gpt-5.4
    python -m evaluation.main --subject-id 10196757 --hadm-id 24725711 --all-models
    python -m evaluation.main --manifest data/cases/manifest.jsonl --model gpt-5.4
    python -m evaluation.main --manifest data/cases/manifest.jsonl --all-models
    python -m evaluation.main --all --model gpt-5.4          # scan all cases in data/cases/
    python -m evaluation.main --all --all-models --process   # full run with process scoring
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

from env.config_loader import get_config
from evaluation.scorer import score_stage
from evaluation.scorers._usage import usage as _usage, progress as _progress
from evaluation.scorers.process_scorers import score_process

_CASES_DIR = Path(__file__).parent.parent / "data" / "cases"


# ------------------------------------------------------------------ #
# I/O helpers                                                          #
# ------------------------------------------------------------------ #

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _model_dirs(case_dir: Path, model: str | None) -> list[tuple[str, Path]]:
    runs_dir = case_dir / "runs"
    if not runs_dir.is_dir():
        return []
    if model:
        d = runs_dir / model
        return [(model, d)] if d.is_dir() and (d / "output.json").exists() else []
    return [
        (d.name, d)
        for d in sorted(runs_dir.iterdir())
        if d.is_dir() and (d / "output.json").exists()
    ]


# ------------------------------------------------------------------ #
# Core: score one (case, model) pair                                  #
# ------------------------------------------------------------------ #

def evaluate_one(
    subject_id:  str,
    hadm_id:     str,
    model_name:  str,
    run_process: bool = False,
    verbose:     bool = True,
) -> dict | None:
    case_dir  = _CASES_DIR / str(subject_id) / str(hadm_id)
    model_dir = case_dir / "runs" / model_name

    output   = _load_json(model_dir / "output.json")
    dialogue = _load_json(model_dir / "dialogue.json")
    case     = _load_json(case_dir  / "case" / "case.json")

    if not output:
        if verbose:
            print(f"  [skip] no output.json: {model_dir / 'output.json'}")
        return None

    eval_conf = get_config("eval")
    proc_conf = eval_conf.get("process_scorers", {})

    _usage.reset()

    out_stages_list = output.get("stages", [])
    diag_stages     = (dialogue or {}).get("stages", [])
    case_stages     = (case     or {}).get("stages", [])

    def _score_one_stage(i: int, out_stage: dict) -> dict:
        outcome = score_stage(out_stage)
        stage_result: dict = {
            "stage_idx":     i,
            "label":         out_stage.get("label", f"stage_{i}"),
            "outcome_score": {
                "total":     outcome["total"],
                "hits":      outcome["hits"],
                "recall":    outcome["recall"],
                "precision": outcome["precision"],
                "f1":        outcome["f1"],
                "score":     outcome["score"],
                "matches":   outcome["matches"],
            },
        }
        if run_process and dialogue and case:
            diag_stage = diag_stages[i] if i < len(diag_stages) else {}
            case_stage = case_stages[i] if i < len(case_stages) else {}
            process    = score_process(diag_stage, case_stage, proc_conf)
            if process:
                stage_result["process_score"] = process
        return stage_result

    scored_stages: list[dict] = [None] * len(out_stages_list)  # type: ignore[list-item]
    if out_stages_list:
        with ThreadPoolExecutor(max_workers=len(out_stages_list)) as stage_pool:
            futs = {stage_pool.submit(_score_one_stage, i, s): i
                    for i, s in enumerate(out_stages_list)}
            for fut in as_completed(futs):
                idx = futs[fut]
                scored_stages[idx] = fut.result()

    # ── Build eval record ─────────────────────────────────────────────
    eval_at = datetime.now(timezone.utc).isoformat()
    eval_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    eval_record = {
        "eval_id":         eval_id,
        "eval_at":         eval_at,
        "eval_model":      eval_conf.get("eval_model", ""),
        "subject_id":      subject_id,
        "hadm_id":         hadm_id,
        "model":           model_name,
        "config_snapshot": eval_conf,
        "usage":           _usage.snapshot(),
        "stages":          scored_stages,
    }

    # ── Persist ───────────────────────────────────────────────────────
    eval_dir = case_dir / "evals" / model_name
    _save_json(eval_dir / f"{eval_id}.json", eval_record)
    _save_json(eval_dir / "latest.json",     eval_record)

    # ── Print summary ─────────────────────────────────────────────────
    if verbose:
        print(f"  {subject_id}/{hadm_id}/{model_name}  [eval_id={eval_id}]")
        for s in scored_stages:
            sc    = s["outcome_score"]
            label = s["label"] or f"stage_{s['stage_idx']}"
            line  = f"    {label:30} F1={sc['f1']:.3f}  R={sc['recall']:.3f}  P={sc['precision']:.3f}"
            proc  = s.get("process_score", {})
            if proc.get("info_coverage"):
                cov = proc["info_coverage"]
                line += f"  cov={cov.get('coverage_overall', 0):.3f}  eff={cov.get('efficiency', 0):.3f}"
            if proc.get("lab_cost"):
                lc      = proc["lab_cost"]
                wasted  = lc.get("wasted_ratio")
                w_str   = f"{wasted:.3f}" if wasted is not None else "null"
                line   += f"  lab_waste={w_str}"
            print(line)
        u = eval_record["usage"]
        if u["total_tokens"] > 0:
            print(f"    tokens: {u['total_tokens']}  "
                  + "  ".join(f"{k}={v['calls']}calls/{v['prompt_tokens']+v['completion_tokens']}tok"
                               for k, v in u["by_scorer"].items()))

    return eval_record


# ------------------------------------------------------------------ #
# Batch helpers                                                        #
# ------------------------------------------------------------------ #

def _cases_from_manifest(manifest_path: Path) -> list[tuple[str, str, str | None]]:
    """Returns list of (subject_id, hadm_id, model_name_or_none)."""
    cases = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            model_name = obj.get("model")  # may be None
            cases.append((str(obj["subject_id"]), str(obj["hadm_id"]), model_name))
    return cases


def _cases_from_dir() -> list[tuple[str, str, None]]:
    """Returns list of (subject_id, hadm_id, None)."""
    pairs = []
    for subj in sorted(_CASES_DIR.iterdir()):
        if not subj.is_dir():
            continue
        for hadm in sorted(subj.iterdir()):
            if hadm.is_dir() and (hadm / "case" / "case.json").exists():
                pairs.append((subj.name, hadm.name, None))
    return pairs


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--subject-id", help="Subject ID (use with --hadm-id)")
    src.add_argument("--manifest",   help="Manifest JSONL path")
    src.add_argument("--all",        action="store_true", help="Scan all cases in data/cases/")

    parser.add_argument("--hadm-id",    help="Admission ID (use with --subject-id)")
    parser.add_argument("--model",      help="Model folder name (default: all found)")
    parser.add_argument("--all-models", action="store_true", help="Evaluate all model dirs")
    parser.add_argument("--process",    action="store_true", help="Run process scorers (LLM calls)")
    parser.add_argument("--workers",    type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--since",      help="Skip cases with eval_id >= this value (e.g. 20260502_204700)")
    parser.add_argument("--skip-process-done", action="store_true",
                        help="Skip cases that already have process_score in latest.json")
    parser.add_argument("--quiet",      action="store_true")

    args    = parser.parse_args()
    verbose = not args.quiet
    model   = None if args.all_models else args.model

    if args.subject_id:
        if not args.hadm_id:
            print("--hadm-id required with --subject-id", file=sys.stderr)
            return 1
        cases = [(args.subject_id, args.hadm_id, None)]
    elif args.manifest:
        cases = _cases_from_manifest(Path(args.manifest))
    else:
        cases = _cases_from_dir()

    # Build work list: (subject_id, hadm_id, model_name)
    work = []
    n_skip = 0
    for subject_id, hadm_id, manifest_model in cases:
        case_dir    = _CASES_DIR / str(subject_id) / str(hadm_id)
        # Use manifest model if specified, otherwise use command-line model
        effective_model = manifest_model or model
        model_pairs = _model_dirs(case_dir, effective_model)
        if not model_pairs:
            n_skip += 1
            continue
        for model_name, _ in model_pairs:
            latest = case_dir / "evals" / model_name / "latest.json"
            # --since: skip if already evaluated after cutoff
            if args.since and latest.exists():
                try:
                    eid = json.loads(latest.read_text()).get("eval_id", "")
                    if eid >= args.since:
                        n_skip += 1
                        continue
                except Exception:
                    pass
            # --skip-process-done: skip if latest.json already has process_score
            if args.skip_process_done and latest.exists():
                try:
                    d = json.loads(latest.read_text())
                    if any(s.get("process_score") for s in d.get("stages", [])):
                        n_skip += 1
                        continue
                except Exception:
                    pass
            work.append((subject_id, hadm_id, model_name))

    # ── Pre-compute total LLM calls for progress tracking ─────────────
    if args.process:
        proc_conf_check = get_config("eval").get("process_scorers", {})
        if proc_conf_check.get("info_coverage", {}).get("enabled"):
            n_speakers = len(proc_conf_check["info_coverage"].get("speakers", ["nurse", "patient", "lab"]))
            total_llm_calls = 0
            for sid, hid, mname in work:
                dial_file = _CASES_DIR / sid / hid / "runs" / mname / "dialogue.json"
                if dial_file.exists():
                    try:
                        n_stages = len(json.loads(dial_file.read_text()).get("stages", []))
                        total_llm_calls += n_stages * n_speakers
                    except Exception:
                        pass
            _progress.reset(total=total_llm_calls)
            if verbose:
                print(f"  progress tracking: {total_llm_calls} LLM calls expected")

    if verbose:
        print(f"Evaluating {len(work)} case(s)  "
              f"model={'all' if model is None else model}  "
              f"process={args.process}  workers={args.workers}  skipped={n_skip}")
        print()

    n_ok = 0
    lock_print = __import__("threading").Lock()

    def _run(item):
        sid, hid, mname = item
        res = evaluate_one(sid, hid, mname, run_process=args.process, verbose=False)
        if verbose and res:
            sc_lines = []
            for s in res.get("stages", []):
                sc = s["outcome_score"]
                label = s["label"] or f"stage_{s['stage_idx']}"
                sc_lines.append(f"    {label:30} F1={sc['f1']:.3f}  R={sc['recall']:.3f}  P={sc['precision']:.3f}")
            with lock_print:
                print(f"  {sid}/{hid}/{mname}")
                for l in sc_lines:
                    print(l)
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run, item): item for item in work}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                n_ok += 1

    if verbose:
        print(f"\nDone.  scored={n_ok}  skipped={n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
