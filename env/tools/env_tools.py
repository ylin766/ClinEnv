"""Environment tools for the model under test.
Progressive disclosure: get_events (compact list) → get_event_detail (full payload).
Structured submission: submit_medication / submit_diagnosis / submit_procedure,
then finalize_decision to end the stage.
"""

from __future__ import annotations
from typing import Any

# ------------------------------------------------------------------ #
# Tool schemas (OpenAI function-calling format)                       #
# ------------------------------------------------------------------ #

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_medication",
            "description": (
                "Submit one medication you would order or continue for this stage. "
                "Use the standard drug name (e.g. 'Aspirin', 'Heparin', 'Metoprolol'). "
                "Call once per medication. You can submit multiple medications."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {
                        "type": "string",
                        "description": "Standard drug name.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief clinical reasoning for this medication.",
                    },
                },
                "required": ["drug_name", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_diagnosis",
            "description": (
                "Submit one diagnosis you have established or strongly suspect. "
                "Use a standard medical term (e.g. 'Unstable angina', "
                "'Coronary atherosclerosis of native coronary artery'). "
                "Call once per diagnosis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis": {
                        "type": "string",
                        "description": "Standard diagnosis name.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief clinical reasoning for this diagnosis.",
                    },
                },
                "required": ["diagnosis", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_procedure",
            "description": (
                "Submit one procedure or intervention you would order. "
                "Use a standard medical term (e.g. 'Left heart cardiac catheterization', "
                "'Coronary artery bypass grafting'). "
                "Call once per procedure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "procedure": {
                        "type": "string",
                        "description": "Standard procedure name.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief clinical reasoning for this procedure.",
                    },
                },
                "required": ["procedure", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_decision",
            "description": (
                "Finalize your decision for this stage. "
                "Call this after you have submitted all required items as specified in the stage instructions. "
                "You will be rejected if required submission types are missing."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SUBMIT_TOOLS = {"submit_medication", "submit_diagnosis", "submit_procedure"}
FINALIZE_TOOL = "finalize_decision"


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _compact_event(ev: dict) -> dict:
    table = ev.get("source_table", "")
    row: dict[str, Any] = {
        "index": ev.get("index"),
        "time":  ev.get("event_time", ev.get("charttime", ev.get("storetime", ""))),
        "table": table,
    }

    if table == "hosp_pharmacy_df":
        row["label"] = ev.get("medication") or f"[{ev.get('proc_type', 'unknown')}]"
        parts = [ev.get("route", ""), ev.get("frequency", ""), ev.get("duration_interval", "")]
        row["summary"] = " | ".join(p for p in parts if p)

    elif table == "hosp_prescriptions_df":
        row["label"] = ev.get("drug", "")
        dose = f"{ev.get('dose_val_rx', '')} {ev.get('dose_unit_rx', '')}".strip()
        row["summary"] = f"{ev.get('prod_strength', '')} | {ev.get('route', '')} | {dose}".strip(" |")

    elif table == "hosp_procedures_icd_df":
        row["label"] = ev.get("long_title_procedure", "")
        row["icd_code"] = ev.get("icd_code", "")

    elif table == "hosp_labevents_df":
        row["label"] = ev.get("label", "")
        if ev.get("value") is not None:
            row["value"] = str(ev["value"])
        if ev.get("valueuom"):
            row["uom"] = ev["valueuom"]
        if "ref_range_lower" in ev and "ref_range_upper" in ev:
            row["ref_range"] = f"{ev['ref_range_lower']}-{ev['ref_range_upper']}"

    elif table == "hosp_microbiologyevents_df":
        row["label"] = ev.get("test_name", ev.get("org_name", ""))
        row["summary"] = ev.get("interpretation", "")

    elif table in ("ehr_chartevents_df", "ehr_datetime_events_df"):
        row["label"] = ev.get("label", "")
        if ev.get("value") is not None:
            row["value"] = str(ev["value"])
        if ev.get("valueuom"):
            row["uom"] = ev["valueuom"]

    elif table in ("ehr_inputevents_df", "ehr_ingredientevents_df"):
        row["label"] = ev.get("label", "")
        if ev.get("amount") is not None:
            row["value"] = str(ev["amount"])
        if ev.get("amountuom"):
            row["uom"] = ev["amountuom"]

    elif table == "ehr_outputevents_df":
        row["label"] = ev.get("label", "")
        if ev.get("value") is not None:
            row["value"] = str(ev["value"])
        if ev.get("valueuom"):
            row["uom"] = ev["valueuom"]

    elif table == "ehr_procedureevents_df":
        row["label"] = ev.get("label", "")
        if ev.get("value") is not None:
            row["value"] = str(ev["value"])

    elif table == "hosp_emar_detail_df":
        row["label"] = ev.get("medication", ev.get("parent_field_ordername", ""))
        row["summary"] = ev.get("action_type", "")

    elif table == "radiology_note":
        row["label"] = ev.get("label", "Radiology Report")
        row["summary"] = (ev.get("text", "")[:120] + "...") if ev.get("text") else ""

    else:
        for key in ("label", "drug", "medication", "test_name"):
            if ev.get(key):
                row["label"] = str(ev[key])
                break
        for key in ("value", "amount"):
            if ev.get(key) is not None:
                row["value"] = str(ev[key])
                break

    return {k: v for k, v in row.items() if v is not None and v != ""}


def _all_visible_events(stage: dict) -> list[dict]:
    return sorted(stage.get("events", []), key=lambda e: e.get("index", 0))


# ------------------------------------------------------------------ #
# Dispatch                                                             #
# ------------------------------------------------------------------ #

def dispatch(name: str, args: dict, stage: dict) -> Any:
    end_idx = stage["index_range"][1]

    if name == "get_events":
        events = _all_visible_events(stage)
        if not events:
            return {"count": 0, "events": [], "note": "No clinical events recorded in this stage."}
        return {"count": len(events), "events": [_compact_event(e) for e in events]}

    elif name == "get_event_detail":
        idx = args.get("index")
        if idx is None:
            return {"error": "index is required"}
        if idx > end_idx:
            return {"error": f"Index {idx} is beyond the visible range for this stage (max {end_idx})."}
        for ev in _all_visible_events(stage):
            if ev.get("index") == idx:
                return ev
        return {"error": f"Event index {idx} not found."}

    elif name == "submit_medication":
        value = args.get("drug_name", "").strip()
        if not value:
            return {"error": "drug_name is required"}
        return {"status": "recorded", "type": "medication", "value": value, "reasoning": args.get("reasoning", "")}

    elif name == "submit_diagnosis":
        value = args.get("diagnosis", "").strip()
        if not value:
            return {"error": "diagnosis is required"}
        return {"status": "recorded", "type": "diagnosis", "value": value, "reasoning": args.get("reasoning", "")}

    elif name == "submit_procedure":
        value = args.get("procedure", "").strip()
        if not value:
            return {"error": "procedure is required"}
        return {"status": "recorded", "type": "procedure", "value": value, "reasoning": args.get("reasoning", "")}

    elif name == "finalize_decision":
        return {"status": "finalized"}

    else:
        return {"error": f"Unknown tool: {name}"}
