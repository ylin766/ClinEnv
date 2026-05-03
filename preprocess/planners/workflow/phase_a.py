"""
Phase A — extract physician decisions from the discharge note.

Input:  AdmissionRecord
Output: list of decision dicts
        [{"id": int, "description": str, "type_hint": "medication|procedure|plan"}, ...]
"""

import json
import re

from preprocess.loaders.ehr_loader import AdmissionRecord
from .helpers import call_llm, load_prompt


def phase_a(client, record: AdmissionRecord, verbose: bool = True, model: str | None = None) -> list:
    if verbose:
        print("\n========== PHASE A ==========")

    # Prefer the full discharge note; fall back to stitching individual sections.
    note_text = next(
        (e.get("text", "") for e in record.timeline
         if e.get("source_table") == "note_df" and e.get("text")),
        None,
    )
    if note_text:
        note_content = note_text[:6000]
    else:
        parts = []
        for sec in ["Chief Complaint", "History of Present Illness",
                    "Brief Hospital Course", "Discharge Diagnosis"]:
            val = record.get_note_section(sec)
            if val:
                parts.append(f"--- {sec} ---\n{val}")
        note_content = "\n\n".join(parts)

    meta = record.get_metadata()
    prompt = (
        load_prompt("phase_a.txt")
        + "\n\n"
        + f"--- Discharge Note ---\n{note_content}\n\n"
        + f"--- Metadata ---\n{json.dumps(meta)}\n"
    )

    kwargs = {}
    if model:
        kwargs["model"] = model
    raw = call_llm(client, [{"role": "user", "content": prompt}], **kwargs) \
              .choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        parsed = json.loads(m.group()) if m else {"decisions": []}

    decisions = parsed.get("decisions", [])
    if verbose:
        print(f"  → {len(decisions)} candidate decisions")
        for d in decisions:
            print(f'    [{d["id"]}] ({d.get("type_hint", "?")}) {d["description"]}')

    return decisions
