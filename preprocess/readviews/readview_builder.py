"""ReadView builder.
Builds per-stage, per-agent data views from an AdmissionRecord and planner output.

For each stage, the readview is cumulative: each agent sees all events from
index 0 up to stage.index_range[1], filtered to the tables relevant to that agent.

Agent table assignments:
  patient  — note sections (CC, HPI) + admission demographics; no timeline events
  nurse    — bedside monitoring, medication administration, procedures, radiology
  lab      — lab results and microbiology

Pharmacy orders and prescriptions are included in the nurse readview for
within-window records (past facts the nurse knows about); GT protection is
enforced via stage boundary — only events with index ≤ stage_end are visible.
"""

from preprocess.loaders.ehr_loader import AdmissionRecord

# ------------------------------------------------------------------ #
# Table → agent routing                                               #
# ------------------------------------------------------------------ #

_NURSE_TABLES = {
    "ehr_chartevents_df",       # vitals and bedside monitoring
    "ehr_datetime_events_df",   # clinical events with datetime
    "ehr_procedureevents_df",   # bedside procedures
    "ehr_inputevents_df",       # IV and fluid inputs
    "ehr_outputevents_df",      # fluid output
    "ehr_ingredientevents_df",  # IV ingredient detail
    "hosp_emar_detail_df",      # medication administration record
    "radiology_note",           # radiology reports
    "hosp_procedures_icd_df",   # ICD procedure records — GT protection via stage boundary
    "hosp_pharmacy_df",         # pharmacy dispensing records (active/discontinued status)
    "hosp_prescriptions_df",    # prescription orders (dose, route, frequency)
                                # Both: GT protection via stage boundary (GT index > stage_end)
                                # Within-window records are past facts the nurse knows about
}

_LAB_TABLES = {
    "hosp_labevents_df",           # laboratory results
    "hosp_microbiologyevents_df",  # microbiology and culture results
}

# ------------------------------------------------------------------ #
# Per-agent readview builders                                         #
# ------------------------------------------------------------------ #

def _build_patient_readview(record: AdmissionRecord) -> dict:
    """
    Patient readview is static across stages: CC, HPI, and demographics.
    The patient presents the same complaint regardless of stage.
    """
    return {
        "chief_complaint":  record.get_note_section("Chief Complaint"),
        "hpi":              record.get_note_section("History of Present Illness"),
        "past_medical_history": record.get_note_section("Past Medical History"),
        "age":              record.admission_meta.get("age"),
        "gender":           record.admission_meta.get("gender"),
    }


def _build_nurse_readview(visible_events: list) -> dict:
    """
    Nurse readview: bedside monitoring, EMAR, procedures, radiology.
    Contains all nurse-table events visible at this stage (cumulative).
    """
    events = [e for e in visible_events if e["source_table"] in _NURSE_TABLES]
    return {"events": events}


def _build_lab_readview(visible_events: list) -> dict:
    """
    Lab readview: lab results and microbiology.
    Contains all lab-table events visible at this stage (cumulative).
    Degenerate entries (label shorter than 2 characters) are excluded.
    """
    events = [
        e for e in visible_events
        if e["source_table"] in _LAB_TABLES
        and len(str(e.get("label") or e.get("test_name") or "")) >= 2
    ]
    return {"events": events}


# ------------------------------------------------------------------ #
# Main entry point                                                     #
# ------------------------------------------------------------------ #

def build_readviews(record: AdmissionRecord, stages: list) -> list:
    """
    Enrich each stage dict with a 'readviews' key containing per-agent views.

    Args:
        record: the loaded AdmissionRecord
        stages: list of stage dicts from the planner (each has 'index_range')

    Returns:
        A new list of stage dicts, each with a 'readviews' sub-dict added.
    """
    patient_rv = _build_patient_readview(record)

    enriched = []
    for stage in stages:
        start_idx, end_idx = stage["context_range"]
        visible = [e for e in record.timeline if e["index"] <= end_idx]
        stage_events = [e for e in record.timeline if start_idx <= e["index"] <= end_idx]

        available_agents = ["patient", "history"]
        if any(e["source_table"] in _NURSE_TABLES for e in stage_events):
            available_agents.append("nurse")
        if any(e["source_table"] in _LAB_TABLES for e in stage_events):
            available_agents.append("lab")

        enriched.append({
            **stage,
            "events": stage_events,          # events within this stage only (used internally)
            "visible_events": visible,        # cumulative [0, end_idx] — used by direct mode prompt
            "available_agents": available_agents,
            "readviews": {
                "patient": patient_rv,
                "nurse":   _build_nurse_readview(visible),
                "lab":     _build_lab_readview(visible),
            },
        })

    return enriched
