"""Planner tool definitions.
OpenAI function-call schemas and dispatcher for the decision planner agent.
Each tool maps to a method on AdmissionRecord, giving the planner read-only
access to the admission data during its exploration phase.
"""

import json
import re
from preprocess.loaders.ehr_loader import AdmissionRecord

# ------------------------------------------------------------------ #
# Tool schemas exposed to the planner LLM                             #
# ------------------------------------------------------------------ #

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_metadata",
            "description": (
                "Return admission-level metadata: admission time, approximate "
                "discharge time, admission type/location, discharge location, "
                "patient age, gender, and date of death if recorded."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_note_section",
            "description": (
                "Extract a named section from the discharge note. "
                "Useful section names: 'Chief Complaint', "
                "'History of Present Illness', 'Past Medical History', "
                "'Brief Hospital Course', 'Discharge Medications', "
                "'Discharge Diagnosis', 'Discharge Instructions', "
                "'Pertinent Results'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_name": {
                        "type": "string",
                        "description": "Exact section header name (without colon).",
                    }
                },
                "required": ["section_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline_index",
            "description": (
                "Return a flat chronological list of all clinical timeline events. "
                "Each entry has: index (int), event_time, source_table, label. "
                "Use this to understand the full event sequence and identify "
                "natural stage boundaries as continuous index ranges."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_by_index",
            "description": (
                "Return a compact summary row for each event with index in "
                "[start, end] (inclusive). Same format as get_timeline_index. "
                "Procedure events include their icd_code in the label. "
                "Use this to survey a stage range without downloading full payloads. "
                "Use this to survey a stage range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end":   {"type": "integer"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_diagnoses",
            "description": (
                "Return all discharge ICD diagnosis records. "
                "These are retrospective codes not visible to the doctor model; "
                "use them to verify ground truth only."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_gt_novelty",
            "description": (
                "Check whether medication/procedure GT items are genuinely NEW decisions "
                "that did not already appear in the stage's visible range [start, end]. "
                "A GT drug that was already prescribed at any index <= stage end is a "
                "daily pharmacy refresh, not a new clinical decision, and must be replaced. "
                "Call this during Phase D with the full proposed stages. "
                "Returns each medication GT item with PASS (first appearance) or FAIL (already visible)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stages": {
                        "type": "array",
                        "description": "Full stages array with label, index_range, and gt items.",
                        "items": {"type": "object"},
                    },
                },
                "required": ["stages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_gt_indices",
            "description": (
                "Mechanically verify all GT index constraints: "
                "(1) every stage has at least one GT item, "
                "(2) no GT item points to a pre-admission event (pre_admission=true), "
                "(3) every GT item with a numeric index has index strictly greater than the stage's end index. "
                "Call this during Phase D self-check with the full proposed stages array. "
                "Returns an authoritative PASS/FAIL per item — do not rely on mental arithmetic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stages": {
                        "type": "array",
                        "description": "Array of stage objects, each with label, index_range [start,end], and gt list.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label":       {"type": "string"},
                                "index_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                                "gt":          {"type": "array", "items": {"type": "object"}},
                            },
                            "required": ["label", "index_range", "gt"],
                        },
                    },
                },
                "required": ["stages"],
            },
        },
    },
]

# ------------------------------------------------------------------ #
# Tool dispatcher                                                      #
# ------------------------------------------------------------------ #

def _validate_gt_novelty(record: "AdmissionRecord", stages: list) -> dict:
    """Check that medication/procedure GT items are not already visible in the stage range."""
    passed = []
    violations = []

    for stage in stages:
        label = stage.get("label", "")
        start, end = stage.get("index_range", [0, 0])
        visible_events = [e for e in record.timeline if e.get("index", 0) <= end]

        for gt in stage.get("gt", []):
            if not isinstance(gt, dict):
                continue  # string GT items — skip novelty check
            gt_type   = gt.get("type", "")
            gt_action = gt.get("action", "start")
            if gt_type not in ("medication", "procedure"):
                continue  # plan and diagnosis have no index to check here
            # For medication, only 'start' actions are novel — stop/change are expected to be visible
            if gt_type == "medication" and gt_action != "start":
                continue

            gt_idx = gt.get("index")
            if gt_idx is None:
                continue

            # Check if the same drug/procedure already appears in visible range
            drug_name = (gt.get("drug") or "").strip().lower()
            icd_code  = (gt.get("icd_code") or "").strip()

            already_seen = False
            first_seen_at = None
            for ev in visible_events:
                ev_idx = ev.get("index", 0)
                if ev_idx >= gt_idx:
                    continue  # only check events BEFORE the GT item itself
                # Match by drug name (pharmacy table) or ICD code (procedure table)
                ev_drug = (ev.get("medication") or ev.get("drug") or "").strip().lower()
                ev_icd  = (ev.get("icd_code") or "").strip()
                # Use word-boundary match to avoid false hits like "ns" matching "insulin"
                drug_match = bool(
                    drug_name and re.search(r"\b" + re.escape(drug_name) + r"\b", ev_drug)
                )
                if drug_match or (icd_code and icd_code == ev_icd):
                    already_seen = True
                    first_seen_at = ev_idx
                    break

            entry = {
                "stage":         label,
                "range":         [start, end],
                "gt_index":      gt_idx,
                "gt_item":       gt.get("drug") or gt.get("display") or str(gt),
                "first_seen_at": first_seen_at,
                "result":        "FAIL" if already_seen else "PASS",
            }
            if already_seen:
                entry["fix"] = (
                    f"This drug/procedure already appears at index {first_seen_at} (visible to the model at stage end {end}). "
                    f"It is a daily pharmacy refresh, not a new decision. "
                    f"Replace with a plan GT describing the restart/continuation decision, "
                    f"or find a drug that appears for the first time after index {end}."
                )
                violations.append(entry)
            else:
                passed.append(entry)

    return {
        "status":     "PASS" if not violations else f"FAIL — {len(violations)} pharmacy-refresh GT(s) found",
        "violations": violations,
        "passed":     passed,
    }


def _validate_gt_indices(record: "AdmissionRecord", stages: list) -> dict:
    """Mechanically check GT indices and stage structure."""
    violations = []
    passed = []
    total_events = len(record.timeline)
    last_event_idx = total_events - 1

    # Build a set of pre-admission indices for fast lookup
    pre_admission_indices = {
        ev["index"] for ev in record.timeline if ev.get("pre_admission")
    }

    # Check -1: every stage must have a valid range (start <= end)
    for stage_i, stage in enumerate(stages):
        label = stage.get("label", "")
        index_range = stage.get("index_range", [0, 0])
        if not (isinstance(index_range, list) and len(index_range) == 2):
            violations.append({
                "stage":  label,
                "check":  "index_range is malformed",
                "result": "FAIL",
                "fix":    f"Stage '{label}' has a malformed index_range. Must be [start, end] with start <= end.",
            })
            continue
        start, end = index_range
        if start > end:
            violations.append({
                "stage":  label,
                "range":  [start, end],
                "check":  f"invalid range: start {start} > end {end}",
                "result": "FAIL",
                "fix": (
                    f"Stage '{label}' has start ({start}) > end ({end}). "
                    f"A stage range must have start <= end. "
                    f"If you want this stage's GT to be the first event the model does not see, "
                    f"set this stage's range so that end = GT_index - 1. "
                    f"If that would make start > end, merge this stage with the preceding stage."
                ),
            })

    # Check 0: every stage must have at least one GT item
    for stage_i, stage in enumerate(stages):
        label = stage.get("label", "")
        gt = stage.get("gt", [])
        if not gt:
            violations.append({
                "stage":  label,
                "check":  "gt list is empty",
                "result": "FAIL",
                "fix": (
                    f"Stage '{label}' has no GT items. Every stage must have at least one GT. "
                    f"Find a significant attending decision after this stage ends: a procedure, "
                    f"significant medication start (anticoagulant, antibiotic, vasopressor, insulin, "
                    f"disease-specific agent), or a plan-type GT from the BHC describing a key decision."
                ),
            })

    # Check 1: every indexed GT item has index > stage end AND is not pre-admission
    for stage_i, stage in enumerate(stages):
        label = stage.get("label", "")
        index_range = stage.get("index_range", [0, 0])
        if not (isinstance(index_range, list) and len(index_range) == 2):
            continue  # already caught by Check -1
        start, end = index_range
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            continue
        for gt in stage.get("gt", []):
            if not isinstance(gt, dict):
                continue  # string GT items have no index to check
            gt_type   = gt.get("type", "")
            gt_action = gt.get("action", "start")
            idx = gt.get("index")
            # Coerce string indices to int (model sometimes outputs "78" instead of 78)
            if idx is not None:
                try:
                    idx = int(idx)
                    gt["index"] = idx  # normalise in-place
                except (TypeError, ValueError):
                    idx = None
            display = gt.get("display") or gt.get("drug") or gt.get("span") or str(gt)

            # procedure and medication(start) MUST have a non-null integer index
            # medication stop/increase_dose/decrease_dose do NOT require an index
            needs_index = gt_type == "procedure" or (gt_type == "medication" and gt_action == "start")
            if idx is None:
                if needs_index:
                    violations.append({
                        "stage":   label,
                        "gt_item": display,
                        "check":   f"{gt_type} GT has no index",
                        "result":  "FAIL",
                        "fix": (
                            f"GT item {display!r} has type='{gt_type}' action='{gt_action}' but no index. "
                            f"'start' medication GT and procedure GT must reference a specific timeline event by index. "
                            f"If this decision has no timeline event, change its type to 'plan' and "
                            f"set section='Brief Hospital Course' with a span capturing only the decision action."
                        ),
                    })
                continue  # plan / diagnosis / non-start medication — no index to check

            # Pre-admission check (takes priority)
            if idx in pre_admission_indices:
                violations.append({
                    "stage":    label,
                    "range":    [start, end],
                    "gt_index": idx,
                    "gt_item":  display,
                    "check":    f"index {idx} is a pre-admission event",
                    "result":   "FAIL",
                    "fix": (
                        f"GT index {idx} ({display!r}) is a pre-admission event "
                        f"(event_time before admission_time, flagged pre_admission=true). "
                        f"Remove it from GT — pre-admission events cannot be new physician decisions "
                        f"during this admission. Find a different GT: look for procedures or "
                        f"significant medications that occur after the admission start."
                    ),
                })
                continue

            entry = {
                "stage":    label,
                "range":    [start, end],
                "gt_index": idx,
                "gt_item":  display,
                "check":    f"{idx} > {end}",
                "result":   "PASS" if idx > end else "FAIL",
            }
            if idx > end:
                passed.append(entry)
            else:
                if idx <= start and stage_i == 0:
                    entry["fix"] = (
                        f"GT index {idx} falls within Stage 1's visible window [0, {end}]. "
                        f"The model sees all events from 0 to {end}, so this GT is already visible. "
                        f"Set Stage 1's end to {idx - 1} so index {idx} is the first unseen event. "
                        f"Stage 1 must start at 0 and end at {idx - 1}."
                    )
                elif idx <= start:
                    entry["fix"] = (
                        f"GT index {idx} is at or before this stage's start ({start}). "
                        f"Reducing end below start would make an invalid range. "
                        f"Move this GT item to the previous stage's gt list (which ends at {start - 1}, before this index). "
                        f"The previous stage already sees events up to {start - 1}, so index {idx} is correctly outside its window."
                    )
                else:
                    entry["fix"] = f"Stage ends at {end} but GT index is {idx}. Set this stage's end to {idx - 1}."
                violations.append(entry)

    # Check 2: final stage must end at total_events-1
    if stages:
        last_end = stages[-1].get("index_range", [0, 0])[1]
        if last_end != last_event_idx:
            violations.append({
                "stage": stages[-1].get("label", ""),
                "check": f"last stage ends at {last_end}, expected {last_event_idx}",
                "result": "FAIL",
                "fix": f"Set the final stage's end index to {last_event_idx} (total_events={total_events}).",
            })
        else:
            passed.append({"check": f"final stage ends at {last_event_idx}", "result": "PASS"})

    # Check 3: stages must be contiguous (no gaps)
    for i in range(1, len(stages)):
        prev_end = stages[i-1].get("index_range", [0, 0])[1]
        curr_start = stages[i].get("index_range", [0, 0])[0]
        if curr_start != prev_end + 1:
            violations.append({
                "stage": stages[i].get("label", ""),
                "check": f"stage starts at {curr_start}, expected {prev_end + 1}",
                "result": "FAIL",
                "fix": f"Stage {i+1} should start at {prev_end + 1} (immediately after previous stage ends at {prev_end}).",
            })

    return {
        "status":        "PASS" if not violations else f"FAIL — {len(violations)} violation(s)",
        "total_events":  total_events,
        "violations":    violations,
        "passed":        passed,
    }


def dispatch(record: AdmissionRecord, name: str, args: dict):
    """Route a tool call name + args to the corresponding AdmissionRecord method."""
    if name == "get_metadata":
        return record.get_metadata()
    if name == "get_note_section":
        return record.get_note_section(args["section_name"])
    if name == "get_timeline_index":
        return record.get_timeline_index()
    if name == "get_events_by_index":
        return record.get_events_by_index(args["start"], args["end"])
    if name == "get_diagnoses":
        return record.get_diagnoses()
    if name == "validate_gt_indices":
        return _validate_gt_indices(record, args["stages"])
    if name == "validate_gt_novelty":
        if "stages" not in args:
            return {"error": "stages parameter is required. Pass the complete stages array including label, index_range, and gt fields."}
        return _validate_gt_novelty(record, args["stages"])
    raise ValueError(f"Unknown tool: {name}")
