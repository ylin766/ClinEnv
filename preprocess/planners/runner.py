"""
Planner entry point with enhanced per-case logging.

Pipeline: Phase A -> Phase B -> Phase B2 -> Phase C -> merge -> Phase D
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load .env from project root
dotenv_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)

from env.llm_client import get_openai_client, get_anthropic_client

from preprocess.loaders.ehr_loader import load_admission
from env.config_loader import get_config
from .workflow import phase_a, phase_b, phase_b2, phase_c, phase_d, merge_small_stages, scan_diagnoses
from .workflow.helpers import DEFAULT_MODEL

DATA_ROOT = str(Path(__file__).parent.parent.parent / "mimic-ext-time-series" / "Merge" / "ehr_by_subject")
MANIFEST = str(Path(__file__).parent.parent.parent / "data" / "env_ready_admissions_p50.jsonl")

def run(subject_id: str, hadm_id: str, verbose: bool = True, model: str | None = None, log_dir: str | None = None) -> list:
    """Run pipeline and capture all output. If log_dir is set, writes to file."""
    conf = get_config("planner")
    model = model or conf.get("model") or DEFAULT_MODEL
    
    # Setup logging if requested
    log_handle = None
    if log_dir:
        log_file = Path(log_dir) / f"{subject_id}_{hadm_id}.log"
        log_handle = open(log_file, "w", encoding="utf-8")
    
    def lprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
        if log_handle:
            print(*args, file=log_handle, **kwargs)
            log_handle.flush()

    lprint(f"=== RUN START: {datetime.now().isoformat()} ===")
    lprint(f"CASE: {subject_id}/{hadm_id}")
    lprint(f"MODEL: {model}")
    lprint("-" * 60)

    try:
        # Client selection
        if "claude" in model.lower():
            client = get_anthropic_client()
        else:
            client = get_openai_client()

        record = load_admission(subject_id, hadm_id, DATA_ROOT)
        lprint(f"Timeline size: {len(record.timeline)} events")

        # Step 0: Scanned Diagnoses
        lprint("\n[STEP 0] Scanning Diagnoses...")
        scanned_diags = scan_diagnoses(client, record, verbose=False, model=model)
        lprint(f"Found {len(scanned_diags)} groundable diagnoses")

        # Step 1: Phase A
        lprint("\n[PHASE A] Extracting Decisions...")
        decisions = phase_a(client, record, verbose=False, model=model)
        for i, d in enumerate(decisions):
            lprint(f"  {i}: [{d.get('type_hint', d.get('type', '?'))}] {d.get('description', '')}")

        # Step 2: Phase B & B2
        lprint("\n[PHASE B/B2] Locating Events...")
        located = phase_b(client, record, decisions, verbose=False, model=model)
        located = phase_b2(client, record, located, verbose=False, model=model)
        for i, l in enumerate(located):
            idx = l.get('event_index') or l.get('event_range', 'N/A')
            lprint(f"  {i}: [{l.get('type_hint', l.get('type', '?'))}] -> anchor={idx}")

        # Step 3: Phase C (Stage Building)
        lprint("\n[PHASE C] Building Stages...")
        stages = phase_c(client, record, located, verbose=False, model=model, scanned_diagnoses=None)
        stages = merge_small_stages(stages, min_events=10, verbose=False)
        stages = phase_c(client, record, [], verbose=False, model=model, scanned_diagnoses=scanned_diags, existing_stages=stages)
        lprint(f"Built {len(stages)} stages")

        # Step 4: Phase D (Classification)
        lprint("\n[PHASE D] Classifying...")
        stages = phase_d(client, stages, record.timeline, verbose=False, model=model)

        lprint("\n" + "="*20 + " FINAL RESULT " + "="*20)
        for i, s in enumerate(stages):
            lprint(f"\nSTAGE {i} (Context: {s['context_range']})")
            for g in s["gts"]:
                loc = g.get("event_index") or g.get("event_range", "Plan/Diag")
                a = f'/{g["action"]}' if "action" in g and g["action"] != "null" else ""
                lprint(f"  - GT: {loc} [{g.get('type')}{a}] {g.get('description')}")
                # Log enriched fields
                if g.get("drug_name"):
                    lprint(f"    Drug: {g['drug_name']} | Dose: {g.get('dose','')} {g.get('dose_unit','')} | Route: {g.get('route','')} | Freq: {g.get('frequency','')}")
                if g.get("icd_code"):
                    lprint(f"    ICD{g.get('icd_version','')}: {g['icd_code']} | {g.get('procedure_title','')}")

        lprint(f"\n=== RUN END: {datetime.now().isoformat()} ===")
        return stages

    except Exception as e:
        lprint("\n" + "!"*20 + " ERROR " + "!"*20)
        import traceback
        lprint(traceback.format_exc())
        return []
    finally:
        if log_handle:
            log_handle.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out_json", default="results_100_sonnet.json")
    args = parser.parse_args()

    # Create logs directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"run_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_dir}")

    # Load Manifest
    with open(MANIFEST) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    
    random.seed(args.seed)
    cases = random.sample(entries, min(args.sample, len(entries)))
    
    import concurrent.futures
    all_results = {}
    
    def _worker(e):
        sid, hid = str(e["subject_id"]), str(e["hadm_id"])
        stages = run(sid, hid, verbose=False, model=args.model, log_dir=log_dir)
        if not stages:
            raise RuntimeError(f"Case {sid}/{hid} returned 0 stages")
        return f"{sid}/{hid}", stages

    max_workers = 10
    print(f"Running {len(cases)} cases with {max_workers} workers...")
    
    consecutive_failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, e): e for e in cases}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                case_id, stages = future.result()
                all_results[case_id] = stages
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                print(f"FAILED: {exc}")
                if consecutive_failures >= 3:
                    print("3 consecutive failures — aborting to save tokens!")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return 1
            done_count += 1
            if done_count % 5 == 0:
                print(f"Completed {done_count}/{len(cases)} cases...")

    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDone! Results saved to {args.out_json}. Logs are in {log_dir}")

if __name__ == "__main__":
    main()
