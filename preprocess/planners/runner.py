"""
Planner entry point with enhanced per-case logging.

Pipeline: Phase A -> Phase B -> Phase C -> Phase D -> Phase E
"""

import json
import os
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
from .workflow import phase_a, phase_b, phase_c, phase_d, phase_e
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

        # Step 0: Phase D (Diagnosis Scanning)
        lprint("\n[PHASE D] Scanning Diagnoses...")
        scanned_diags = phase_d(client, record, verbose=False, model=model)
        lprint(f"Found {len(scanned_diags)} groundable diagnoses")

        # Step 1: Phase A
        lprint("\n[PHASE A] Extracting Decisions...")
        decisions = phase_a(client, record, verbose=False, model=model)
        for i, d in enumerate(decisions):
            lprint(f"  {i}: [{d.get('type_hint', d.get('type', '?'))}] {d.get('description', '')}")

        # Step 2: Phase B
        lprint("\n[PHASE B] Locating Events...")
        located = phase_b(client, record, decisions, verbose=False, model=model)
        for i, l in enumerate(located):
            idx = l.get('index')
            if idx is None:
                idx = l.get('index_range', 'N/A')
            lprint(f"  {i}: [{l.get('type_hint', l.get('type', '?'))}] -> anchor={idx}")

        # Step 3: Phase C (Stage Building)
        lprint("\n[PHASE C] Building Stages...")
        stages = phase_c(client, record, located, verbose=False, model=model, scanned_diagnoses=None)
        stages = phase_c(client, record, [], verbose=False, model=model, scanned_diagnoses=scanned_diags, existing_stages=stages)
        lprint(f"Built {len(stages)} stages")

        # Step 4: Phase E (Classification)
        lprint("\n[PHASE E] Classifying...")
        stages = phase_e(client, stages, record.timeline, verbose=False, model=model)

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

