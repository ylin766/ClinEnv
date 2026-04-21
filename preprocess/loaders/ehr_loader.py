"""Raw EHR loader.
Loads a single admission JSON and exposes planner-facing query tools.
Separates metadata (diagnoses, admission record, discharge note) from the
clinical timeline (all other events, ordered by their original array index).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

# Fields used to summarize each table in get_timeline_index()
_TABLE_LABEL_FIELD = {
    "hosp_labevents_df":       "label",
    "hosp_prescriptions_df":   "drug",
    "hosp_pharmacy_df":        "medication",
    "hosp_procedures_icd_df":  "long_title_procedure",
    "hosp_microbiologyevents_df": "test_name",
    "hosp_emar_detail_df":     "medication",
    "radiology_note":          None,   # free text, summarised differently
    "ehr_chartevents_df":      "label",
    "ehr_datetime_events_df":  "label",
    "ehr_ingredientevents_df": "label",
    "ehr_inputevents_df":      "label",
    "ehr_outputevents_df":     "label",
    "ehr_procedureevents_df":  "label",
}

# Tables treated as metadata: never enter the clinical timeline
_METADATA_TABLES = {"hosp_diagnoses_icd_df", "hosp_admissions_df", "note_df"}


@dataclass
class AdmissionRecord:
    subject_id: str
    hadm_id: str

    # metadata: planner-only, never shown to doctor model via readview
    admission_meta: dict = field(default_factory=dict)   # hosp_admissions_df fields
    diagnoses: list = field(default_factory=list)        # hosp_diagnoses_icd_df records
    discharge_note: Optional[str] = None                 # note_df full text

    # clinical timeline: ordered by original array index
    # each entry: {"index": int, "source_table": str, "event_time": str|None, ...fields}
    timeline: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Planner tools                                                        #
    # ------------------------------------------------------------------ #

    def get_metadata(self) -> dict:
        """Return admission-level metadata: timing, demographics, discharge fields."""
        admission_time = self.admission_meta.get("event_time")
        last_event = self.timeline[-1] if self.timeline else {}
        discharge_time_approx = last_event.get("event_time")

        return {
            "admission_time":          admission_time,
            "discharge_time_approx":   discharge_time_approx,  # last timeline event
            "admission_type":          self.admission_meta.get("admission_type"),
            "admission_location":      self.admission_meta.get("admission_location"),
            "discharge_location":      self.admission_meta.get("discharge_location"),
            "age":                     self.admission_meta.get("age"),
            "gender":                  self.admission_meta.get("gender"),
            "dod":                     self.admission_meta.get("dod"),
        }

    def get_notes(self) -> Optional[str]:
        """Return the discharge note full text."""
        return self.discharge_note

    def get_diagnoses(self) -> list[dict]:
        """Return all discharge ICD diagnosis records (metadata, GT use only)."""
        return self.diagnoses

    def get_timeline_index(self) -> list[dict]:
        """
        Return a flat chronological list of all timeline events, one entry per
        event.  Each entry is compact: index, event_time, source_table, and a
        short human-readable label derived from the event's key field.

        This gives the planner a complete ordered view of the admission so it
        can identify natural stage boundaries as continuous index ranges
        [start, end] that cover all events belonging to that stage.
        """
        rows = []
        for ev in self.timeline:
            table = ev["source_table"]
            label_field = _TABLE_LABEL_FIELD.get(table)

            if label_field:
                label = ev.get(label_field, "")
                # for lab events also attach value + flag if present
                if table == "hosp_labevents_df":
                    val  = ev.get("value", "")
                    flag = ev.get("flag", "")
                    if val:
                        label = f"{label}: {val}"
                    if flag:
                        label = f"{label} [{flag}]"
            elif table == "radiology_note":
                label = (ev.get("text") or "")[:80].replace("\n", " ").strip()
            else:
                label = ""

            rows.append({
                "index":        ev["index"],
                "event_time":   ev.get("event_time"),
                "source_table": table,
                "label":        label,
            })

        return rows

    def get_note_section(self, section_name: str) -> Optional[str]:
        """
        Extract a named section from the discharge note.
        Common section names: 'Chief Complaint', 'History of Present Illness',
        'Past Medical History', 'Brief Hospital Course', 'Discharge Medications',
        'Discharge Diagnosis', 'Discharge Instructions', 'Pertinent Results'.
        Returns the section text, or None if the section is not found.
        """
        if not self.discharge_note:
            return None
        note = self.discharge_note
        # section headers end with ':' and are followed by a newline
        start_marker = section_name + ":"
        start = note.find(start_marker)
        if start == -1:
            return None
        start = note.index("\n", start) + 1  # skip the header line itself

        # find the next section header (a line ending with ':')
        import re
        next_header = re.search(r"\n[A-Z][^\n]+:\n", note[start:])
        end = start + next_header.start() if next_header else len(note)
        return note[start:end].strip()

    def get_events_by_index(self, start: int, end: int) -> list[dict]:
        """
        Return a compact summary row for each event with index in [start, end].
        Same format as get_timeline_index but filtered to the given range.
        For procedures and diagnoses also includes the icd_code.
        Use get_event_detail(index) to retrieve the full payload of a specific event.
        """
        rows = []
        for ev in self.timeline:
            if not (start <= ev["index"] <= end):
                continue
            table = ev["source_table"]
            label_field = _TABLE_LABEL_FIELD.get(table)

            if label_field:
                label = ev.get(label_field, "")
                if table == "hosp_labevents_df":
                    val  = ev.get("value", "")
                    flag = ev.get("flag", "")
                    if val:
                        label = f"{label}: {val}"
                    if flag:
                        label = f"{label} [{flag}]"
                elif table == "hosp_procedures_icd_df":
                    icd = ev.get("icd_code", "")
                    ver = ev.get("icd_version", "")
                    label = f"[ICD{ver}:{icd}] {label}"
            elif table == "radiology_note":
                label = (ev.get("text") or "")[:80].replace("\n", " ").strip()
            else:
                label = ""

            rows.append({
                "index":        ev["index"],
                "event_time":   ev.get("event_time"),
                "source_table": table,
                "label":        label,
            })
        return rows

    def get_event_detail(self, index: int) -> dict:
        """
        Return the full payload for a single timeline event by its index.
        Use this after get_events_by_index or search_events to retrieve
        the complete fields of one specific event.
        """
        for ev in self.timeline:
            if ev["index"] == index:
                return ev
        return {}

    def search_events(self, keyword: str) -> list[dict]:
        """
        Case-insensitive keyword search across all timeline event field values.
        Returns compact summary rows for matching events (same format as
        get_events_by_index). Use get_event_detail(index) for full payload.
        """
        kw = keyword.lower()
        rows = []
        for ev in self.timeline:
            if not any(kw in str(v).lower() for v in ev.values()):
                continue
            table = ev["source_table"]
            label_field = _TABLE_LABEL_FIELD.get(table)
            if label_field:
                label = ev.get(label_field, "")
                if table == "hosp_procedures_icd_df":
                    icd = ev.get("icd_code", "")
                    ver = ev.get("icd_version", "")
                    label = f"[ICD{ver}:{icd}] {label}"
            elif table == "radiology_note":
                label = (ev.get("text") or "")[:80].replace("\n", " ").strip()
            else:
                label = ""
            rows.append({
                "index":        ev["index"],
                "event_time":   ev.get("event_time"),
                "source_table": table,
                "label":        label,
            })
        return rows


# ------------------------------------------------------------------ #
# Loader entry point                                                   #
# ------------------------------------------------------------------ #

def load_admission(subject_id: str, hadm_id: str, data_root: str) -> AdmissionRecord:
    """
    Load a single admission JSON file and return an AdmissionRecord.

    Args:
        subject_id: patient subject identifier
        hadm_id:    hospital admission identifier
        data_root:  path to ehr_by_subject directory
    """
    path = os.path.join(data_root, str(subject_id), f"{hadm_id}.json")
    with open(path, encoding="utf-8") as f:
        raw_events: list[dict] = json.load(f)

    record = AdmissionRecord(subject_id=str(subject_id), hadm_id=str(hadm_id))
    timeline_index = 0

    for raw in raw_events:
        table = raw.get("source_table", "")

        if table == "hosp_admissions_df":
            record.admission_meta = {k: v for k, v in raw.items()
                                     if k != "source_table"}

        elif table == "hosp_diagnoses_icd_df":
            record.diagnoses.append({k: v for k, v in raw.items()
                                     if k != "source_table"})

        elif table == "note_df":
            record.discharge_note = raw.get("text")

        else:
            event = {"index": timeline_index, **raw}
            record.timeline.append(event)
            timeline_index += 1

    return record
