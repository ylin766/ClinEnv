"""ReadView builder.
Builds per-stage, per-agent data views from an AdmissionRecord and planner output.

For each stage, the readview is cumulative: each agent sees all events from
index 0 up to stage.index_range[1], filtered to the tables relevant to that agent.

Agent table assignments:
  patient  — note sections (CC, HPI) + admission demographics; no timeline events
  nurse    — bedside monitoring, medication administration, procedures, radiology
  lab      — lab results and microbiology

Tables that are physician-decision records (pharmacy orders, ICD procedures,
prescriptions) are NOT included in any readview — they are used for GT only.
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
}

_LAB_TABLES = {
    "hosp_labevents_df",           # laboratory results
    "hosp_microbiologyevents_df",  # microbiology and culture results
}

# Tables not routed to any agent readview (physician-decision records / GT only)
_PHYSICIAN_ONLY_TABLES = {
    "hosp_pharmacy_df",
    "hosp_prescriptions_df",
    "hosp_procedures_icd_df",
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
    """
    events = [e for e in visible_events if e["source_table"] in _LAB_TABLES]
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
        end_idx = stage["index_range"][1]
        visible = [e for e in record.timeline if e["index"] <= end_idx]

        enriched.append({
            **stage,
            "readviews": {
                "patient": patient_rv,
                "nurse":   _build_nurse_readview(visible),
                "lab":     _build_lab_readview(visible),
            },
        })

    return enriched
