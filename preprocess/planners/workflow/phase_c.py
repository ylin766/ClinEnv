"""
Phase C — assign context windows for each decision.

For each decision located in Phase B/B2:
  - Compute the available context window [prev_end+1, decision_start-1]
  - Reject medication GTs where the same drug is already actively visible in context
  - Skip decisions with no context window (inherited by next stage)

Input:  located decisions from Phase B2 (with `index` or `index_range`)
Output: list of stage dicts
        [{"context_range": [start, end], "gt": {...}}, ...]
"""

import json
import re

from preprocess.loaders.ehr_loader import AdmissionRecord
from ..tools import _event_drug_tokens, _drug_tokens_from_description
from .helpers import call_llm, fmt_events, load_prompt


def phase_c(client, record: AdmissionRecord, decisions: list,
            verbose: bool = True, model: str | None = None,
            scanned_diagnoses: list = None,
            existing_stages: list = None) -> list:
    if existing_stages is not None:
        stages = existing_stages
    else:
        if verbose:
            print("\n========== PHASE C ==========")
        stages = []

    total = len(record.timeline)
    llm_kwargs = {"model": model} if model else {}

    def get_decision_start(d):
        if d.get("index_range"):
            return d["index_range"][0]
        if d.get("index") is not None:
            return d["index"]
        return None

    def get_decision_end(d):
        if d.get("index_range"):
            return d["index_range"][1]
        if d.get("index") is not None:
            return d["index"]
        return None

    def build_gt(d):
        if d.get("index_range"):
            return {"type": d.get("type_hint", "procedure"),
                    "event_range": d["index_range"],
                    "description": d["description"]}
        return {"type": d.get("type_hint", "medication"),
                "event_index": d["index"],
                "description": d["description"]}

    # ---- Build stages from located decisions ----
    if existing_stages is None:
        prev_end = -1

        for decision in decisions:
            d_start = get_decision_start(decision)
            if d_start is None:
                # Decision was not located on timeline — skip entirely
                if verbose:
                    print(f'  [C] decision {decision["id"]} has no anchor — skipped')
                continue

            avail_start = prev_end + 1
            context_end = d_start - 1

            if avail_start > context_end:
                if verbose:
                    print(f'  [C] decision {decision["id"]} at {d_start} — no context window, skip')
                # Don't update prev_end: context inherited by next stage
                continue

            # Reject medication GTs where the same drug is already actively in context
            if decision.get("type_hint") == "medication":
                gt_drug_tokens = _drug_tokens_from_description(decision.get("description", ""))
                _stopped = ("inactive", "expired", "canceled", "cancelled")
                already_active = False
                for ci in range(avail_start, context_end + 1):
                    evt = record.timeline[ci]
                    if evt.get("source_table") not in ("hosp_pharmacy_df", "hosp_prescriptions_df"):
                        continue
                    evt_tokens = _event_drug_tokens(evt)
                    if not (gt_drug_tokens & evt_tokens):
                        continue
                    status = evt.get("status", "").lower()
                    is_stopped = (
                        any(s in status for s in _stopped)
                        or ("discontinued" in status and "via patient discharge" not in status)
                    )
                    if not is_stopped:
                        already_active = True
                        break
                if already_active:
                    if verbose:
                        print(f'  [C] decision {decision["id"]} SKIP — drug already active in context')
                    # Don't update prev_end: context inherited by next stage
                    continue

            if verbose:
                print(f'  [C] decision {decision["id"]} at {d_start} → context=[{avail_start},{context_end}]')
            stages.append({
                "context_range": [avail_start, context_end],
                "gt": build_gt(decision),
            })
            prev_end = get_decision_end(decision)

    # ---- Append final diagnosis stage ----
    if scanned_diagnoses and total > 0:
        last_idx = -1
        if stages:
            for s in stages:
                cr = s["context_range"]
                last_idx = max(last_idx, cr[1])
                gt_obj = s.get("gt", {})
                if isinstance(gt_obj, dict):
                    if "event_index" in gt_obj:
                        last_idx = max(last_idx, gt_obj["event_index"])
                    if "event_range" in gt_obj:
                        last_idx = max(last_idx, gt_obj["event_range"][1])

        final_ctx_start = last_idx + 1
        final_ctx_end = total - 1

        if final_ctx_start < total:
            if verbose:
                print(f"  [C] Adding final diagnosis stage: context=[{final_ctx_start},{final_ctx_end}]")
            stages.append({
                "context_range": [final_ctx_start, final_ctx_end],
                "gt": {"type": "diagnosis_list", "diagnoses": scanned_diagnoses}
            })

    if verbose:
        print(f"  → {len(stages)} stages built")

    return stages
