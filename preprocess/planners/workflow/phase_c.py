"""
Phase C — assign context windows for each decision.

For each decision located in Phase B/B2:
  - Bundle decisions that share the same event_time into a single stage GT list
  - Compute the available context window [prev_end+1, decision_start-1]
  - Reject medication GTs where the same drug is already actively visible in context
  - Skip decisions with no context window (inherited by next stage)

Input:  located decisions from Phase B2 (with `index` or `index_range`)
Output: list of stage dicts
        [{"context_range": [start, end], "gts": [...], ...}
"""

import json
import re

from preprocess.loaders.ehr_loader import AdmissionRecord
from ..tools import _event_drug_tokens, _drug_tokens_from_description
from .helpers import call_llm, fmt_events, load_prompt


MIN_CONTEXT_SPAN = 3  # stages with context span < this are bundled into next


def _bundle_decisions(decisions: list, timeline: list) -> list:
    """
    Merge decisions that share the same event_time (or identical index) into
    a single bundle-decision so they get one shared context window.

    Returns a new list where each element is either an original decision dict
    (unchanged) or a dict with key 'bundle' containing a list of decisions
    and 'index'/'index_range' reflecting the full span of the bundle.
    """
    if not decisions:
        return decisions

    def _ts(d):
        """Return event_time string for a located decision, or None."""
        idx = d.get("index")
        if idx is not None and idx < len(timeline):
            return timeline[idx].get("event_time")
        ir = d.get("index_range")
        if ir and ir[0] < len(timeline):
            return timeline[ir[0]].get("event_time")
        return None

    def _start(d):
        if d.get("index") is not None:
            return d["index"]
        if d.get("index_range"):
            return d["index_range"][0]
        return None

    def _end(d):
        if d.get("index") is not None:
            return d["index"]
        if d.get("index_range"):
            return d["index_range"][1]
        return None

    # Group consecutive decisions with same event_time AND adjacent indices (gap <= 5)
    # The gap constraint prevents merging decisions on the same calendar day but
    # widely separated in the timeline (MIMIC pharmacy timestamps are day-level).
    MAX_INDEX_GAP = 5
    groups = []
    current_group = [decisions[0]]
    current_ts = _ts(decisions[0])

    for d in decisions[1:]:
        ts = _ts(d)
        prev_end = _end(current_group[-1])
        cur_start = _start(d)
        adjacent = (prev_end is not None and cur_start is not None
                    and cur_start - prev_end <= MAX_INDEX_GAP)
        if ts is not None and ts == current_ts and cur_start is not None and adjacent:
            current_group.append(d)
        else:
            groups.append(current_group)
            current_group = [d]
            current_ts = ts
    groups.append(current_group)

    result = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
            continue
        # Build a bundle: spans from earliest to latest index
        starts = [s for d in group for s in [_start(d)] if s is not None]
        ends   = [e for d in group for e in [_end(d)]   if e is not None]
        if not starts:
            result.extend(group)
            continue
        bundle = {
            "bundle": group,
            "index_range": [min(starts), max(ends)],
            "type_hint": group[0].get("type_hint", "medication"),
        }
        result.append(bundle)
    return result


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

        # Merge same-timestamp decisions into bundles first
        bundled = _bundle_decisions(
            [d for d in decisions if get_decision_start(d) is not None or d.get("index") is None],
            record.timeline,
        )

        for decision in bundled:
            is_bundle = "bundle" in decision

            d_start = get_decision_start(decision)
            if d_start is None:
                if not is_bundle:
                    if verbose:
                        print(f'  [C] decision {decision.get("id", "?")} has no anchor — skipped')
                continue

            avail_start = prev_end + 1
            context_end = d_start - 1

            if avail_start > context_end:
                label = f'bundle@{d_start}' if is_bundle else f'decision {decision["id"]} at {d_start}'
                if verbose:
                    print(f'  [C] {label} — no context window, skip')
                # Don't update prev_end: context inherited by next stage
                continue

            if is_bundle:
                # Bundle: build multiple GTs for same context window
                gts = [build_gt(d) for d in decision["bundle"]]
                if verbose:
                    ids = [d["id"] for d in decision["bundle"]]
                    print(f'  [C] bundle {ids} → context=[{avail_start},{context_end}] ({len(gts)} GTs)')
                stages.append({
                    "context_range": [avail_start, context_end],
                    "_multi_gt": gts,
                })
                prev_end = get_decision_end(decision)
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
