"""Nurse agent — LLM-backed.

Reports vitals, monitoring data, and bedside observations to the physician
using only information from the nurse readview.
"""

from __future__ import annotations
import json
from pathlib import Path

from openai import OpenAI

_PROMPT = Path(__file__).parent.parent.parent / "prompts" / "env" / "nurse_agent.txt"

# Fields to surface per source table
_TABLE_SUMMARY: dict[str, list[str]] = {
    "ehr_chartevents_df":       ["label", "value", "valueuom", "event_time"],
    "ehr_datetime_events_df":   ["label", "value", "event_time"],
    "ehr_inputevents_df":       ["label", "amount", "amountuom", "event_time"],
    "ehr_outputevents_df":      ["label", "value", "valueuom", "event_time"],
    "ehr_ingredientevents_df":  ["label", "amount", "amountuom", "event_time"],
    "ehr_procedureevents_df":   ["label", "value", "event_time"],
    "hosp_emar_detail_df":      ["medication", "administration_type", "event_txt", "event_time"],
    "radiology_note":           ["text"],
    "hosp_procedures_icd_df":   ["long_title_procedure", "icd_code", "event_time"],
}


def _format_events(events: list[dict]) -> str:
    if not events:
        return "No bedside data available."
    lines = []
    for ev in events:
        table  = ev.get("source_table", "")
        fields = _TABLE_SUMMARY.get(table, ["label", "value", "event_time"])
        parts  = []
        for f in fields:
            val = ev.get(f)
            if val is None:
                continue
            # truncate long text fields (e.g. radiology reports) to keep context manageable
            s = str(val)
            parts.append(s[:400] + "…" if len(s) > 400 else s)
        time   = ev.get("event_time", "")
        prefix = f"[{time}] " if time else ""
        lines.append(prefix + " | ".join(parts))
    return "\n".join(lines)


def answer(readview: dict, query: str, client: OpenAI, model: str) -> str:
    """Return the nurse's response to a physician query."""
    system     = _PROMPT.read_text(encoding="utf-8").strip()
    events_ctx = _format_events(readview.get("events", []))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"{system}\n\n---\nBedside Data:\n{events_ctx}"},
            {"role": "user",   "content": query},
        ],
        temperature=0.3,
        max_completion_tokens=400,
    )
    return resp.choices[0].message.content.strip()
