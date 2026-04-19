"""Prior admissions loader.

For a given subject, finds all admissions that preceded the current one
and extracts the two-level disclosure structure used by the history tools:
  Level 1 (summary): admittime, chief_complaint, bhc, discharge_diagnosis
  Level 2 (detail) : full discharge note text
"""

from __future__ import annotations
import json
import re
from pathlib import Path


def _note_section(note: str, section: str) -> str | None:
    """Extract a named section from a discharge note (same logic as AdmissionRecord)."""
    if not note:
        return None
    start = note.find(section + ":")
    if start == -1:
        return None
    start = note.index("\n", start) + 1
    nxt = re.search(r"\n[A-Z][^\n]+:\n", note[start:])
    end = start + nxt.start() if nxt else len(note)
    return note[start:end].strip() or None


def load_prior_admissions(
    subject_id: str,
    current_hadm_id: str,
    current_admittime: str,
    data_root: str,
) -> list[dict]:
    """Return prior admissions for subject_id, sorted oldest-first.

    Each entry:
        hadm_id, admittime, dischtime,
        chief_complaint, bhc, discharge_diagnosis,  ← level-1 summary
        full_note                                    ← level-2 detail
    """
    subject_dir = Path(data_root) / str(subject_id)
    if not subject_dir.exists():
        return []

    prior: list[dict] = []

    for json_file in subject_dir.glob("*.json"):
        hadm_id = json_file.stem
        if hadm_id == str(current_hadm_id):
            continue

        events: list[dict] = json.loads(json_file.read_text(encoding="utf-8"))
        admittime = dischtime = note_text = None

        for ev in events:
            table = ev.get("source_table", "")
            if table == "hosp_admissions_df":
                admittime = ev.get("admittime") or ev.get("event_time")
                dischtime = ev.get("dischtime")
            elif table == "note_df":
                note_text = ev.get("text")

        # keep only admissions strictly before the current one
        if not admittime or admittime >= current_admittime:
            continue

        prior.append({
            "hadm_id":             hadm_id,
            "admittime":           admittime,
            "dischtime":           dischtime,
            "chief_complaint":     _note_section(note_text, "Chief Complaint"),
            "bhc":                 _note_section(note_text, "Brief Hospital Course"),
            "discharge_diagnosis": _note_section(note_text, "Discharge Diagnosis"),
            "full_note":           note_text,
        })

    prior.sort(key=lambda x: x["admittime"])
    return prior
