"""Preprocessing pipeline entry point.
Connects: load raw EHR → plan decision stages → build readviews → validate → write.

Usage:
    python -m preprocess.main --subject-id 10459005 --hadm-id 22645723
    python -m preprocess.main --manifest data/env_ready_admissions_p50.jsonl --limit 100
"""

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

from preprocess.loaders.ehr_loader import load_admission
from preprocess.loaders.prior_admissions import load_prior_admissions
from preprocess.planners import plan_decision_points
from preprocess.readviews.readview_builder import build_readviews
from preprocess.writers.prepared_case_writer import write_plan, write_prepared_case, update_manifest

DATA_ROOT = str(Path(__file__).parent.parent / "mimic-ext-time-series" / "Merge" / "ehr_by_subject")
OUT_DIR   = Path(__file__).parent.parent / "data" / "cases"


def process_one(subject_id: str, hadm_id: str, verbose: bool = True, model: str | None = None, reuse_plan: bool = False) -> dict:
    """
    Run the full preprocessing pipeline for one admission.
    Returns the prepared case dict.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing {subject_id}/{hadm_id}")
        print(f"{'='*60}")

    # 1. Load raw EHR
    record = load_admission(subject_id, hadm_id, DATA_ROOT)
    if verbose:
        print(f"Loaded {len(record.timeline)} timeline events")

    # 2. Plan decision stages
    plan_path = OUT_DIR / str(subject_id) / str(hadm_id) / "plan" / "plan.json"
    if reuse_plan and plan_path.exists():
        if verbose:
            print(f"Reusing existing plan: {plan_path}")
        stages = json.loads(plan_path.read_text(encoding="utf-8"))["stages"]
        # Backwards compatibility with old plan format
        for s in stages:
            if "index_range" in s:
                s["context_range"] = s.pop("index_range")
            if "gt" in s:
                s["gts"] = s.pop("gt")
            
            # Map old fields to new schema
            for g in s.get("gts", []):
                if "drug" in g:
                    g["drug_name"] = g.pop("drug")
            
    else:
        planner_out = plan_decision_points(record, verbose=verbose, model=model)
        stages = planner_out["stages"]
        if verbose:
            print(f"Planner produced {len(stages)} stages")
        # 3. Save planner output before readviews are added
        write_plan({"subject_id": subject_id, "hadm_id": hadm_id, "stages": stages}, OUT_DIR)

    # 4. Build readviews
    enriched_stages = build_readviews(record, stages)

    # Strip planner-internal fields not needed at runtime
    for stage in enriched_stages:
        stage.pop("decision_required", None)

    # 5. Load prior admissions (for interactive mode history tools)
    current_admittime = record.admission_meta.get("event_time", "")
    prior_admissions  = load_prior_admissions(subject_id, hadm_id, current_admittime, DATA_ROOT)
    if verbose:
        print(f"Found {len(prior_admissions)} prior admission(s)")

    # 6. Assemble prepared case
    prepared: dict = {
        "subject_id":       subject_id,
        "hadm_id":          hadm_id,
        "stages":           enriched_stages,
    }
    if prior_admissions:
        prepared["prior_admissions"] = prior_admissions

    # 7. Write full prepared case to disk
    out_path = write_prepared_case(prepared, OUT_DIR)
    update_manifest(prepared, OUT_DIR)
    if verbose:
        print(f"Written to {out_path}")

    return prepared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject-id", help="Single subject_id (use with --hadm-id)")
    group.add_argument("--manifest",   help="JSONL manifest file to process in batch")
    parser.add_argument("--hadm-id",   help="Single hadm_id (use with --subject-id)")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max number of cases to process from manifest")
    parser.add_argument("--model",     default=None,
                        help="Override planner model (default: gpt-5.4-mini-2026-03-17)")
    parser.add_argument("--reuse-plan", action="store_true", help="Reuse existing plan.json if available")
    parser.add_argument("--quiet",     action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    verbose = not args.quiet
    errors = []

    if args.subject_id:
        if not args.hadm_id:
            print("--hadm-id is required with --subject-id", file=sys.stderr)
            return 1
        try:
            process_one(args.subject_id, args.hadm_id, verbose=verbose, model=args.model, reuse_plan=args.reuse_plan)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1

    else:
        manifest = Path(args.manifest)
        with manifest.open(encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]

        if args.limit:
            records = records[:args.limit]

        from env.config_loader import get_config
        concurrency = get_config("planner").get("concurrency", 8)
        total = len(records)
        print(f"Processing {total} cases from {manifest.name} (concurrency={concurrency})")

        def _run_one(args_tuple):
            idx, rec = args_tuple
            sid, hid = rec["subject_id"], rec["hadm_id"]
            print(f"[{idx}/{total}] {sid}/{hid}")
            process_one(sid, hid, verbose=verbose, model=args.model, reuse_plan=args.reuse_plan)
            return sid, hid, None

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_run_one, (i, rec)): rec for i, rec in enumerate(records, 1)}
            for fut in as_completed(futures):
                rec = futures[fut]
                sid, hid = rec["subject_id"], rec["hadm_id"]
                try:
                    fut.result()
                except Exception as e:
                    msg = f"FAILED {sid}/{hid}: {e}"
                    print(msg, file=sys.stderr)
                    errors.append(msg)

        print(f"\nDone. {total - len(errors)}/{total} succeeded.")
        if errors:
            print(f"{len(errors)} failures:")
            for e in errors:
                print(f"  {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
