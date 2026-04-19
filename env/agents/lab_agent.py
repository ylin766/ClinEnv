"""Lab agent — data lookup (no LLM).

Searches the lab readview for tests matching the doctor's request
and returns structured results. If a test has not been ordered or
resulted, returns not-available.
"""

from __future__ import annotations


def _label(ev: dict) -> str:
    return str(ev.get("label") or ev.get("test_name") or "")


def _format_result(ev: dict) -> dict:
    table = ev.get("source_table", "")
    if table == "hosp_labevents_df":
        result: dict = {
            "test":  _label(ev),
            "value": ev.get("value"),
            "uom":   ev.get("valueuom"),
            "flag":  ev.get("flag"),
            "time":  ev.get("event_time"),
        }
        lo, hi = ev.get("ref_range_lower"), ev.get("ref_range_upper")
        if lo is not None and hi is not None:
            result["ref_range"] = f"{lo}–{hi}"
    else:  # hosp_microbiologyevents_df
        result = {
            "test":           _label(ev),
            "organism":       ev.get("org_name"),
            "interpretation": ev.get("interpretation"),
            "time":           ev.get("event_time"),
        }
    return {k: v for k, v in result.items() if v is not None}


def search(readview: dict, test_name: str) -> dict:
    """Search lab readview for test_name (case-insensitive substring match).

    Matches when either:
      - query is a substring of the label  (e.g. "glucose" → "Glucose, Blood")
      - label is a substring of the query  (e.g. "Hemoglobin" → "Hemoglobin, Blood")
        (label must be ≥3 chars to avoid single-char MIMIC data artefacts)

    Returns:
        {"found": True,  "results": [...]}   if matched
        {"found": False, "message": "..."}   if not found
    """
    kw      = test_name.lower().strip()
    events  = readview.get("events", [])
    matches = [
        ev for ev in events
        if (lab := _label(ev).lower()) and (
            kw in lab or (len(lab) >= 3 and lab in kw)
        )
    ]
    if not matches:
        return {
            "found":   False,
            "message": f"No results available for '{test_name}'. "
                       "The test was not ordered or did not result for this patient.",
        }
    return {
        "found":   True,
        "results": [_format_result(ev) for ev in matches],
    }
