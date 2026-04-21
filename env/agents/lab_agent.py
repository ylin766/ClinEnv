"""Lab agent — LLM-backed lookup.

The physician states what they want in natural language.
The agent uses an LLM to decide which available test labels match the request,
then returns the corresponding results.

If no matching results exist, returns not-available.
"""

from __future__ import annotations
import json
from pathlib import Path

from openai import OpenAI

_PROMPT = Path(__file__).parent.parent.parent / "prompts" / "env" / "lab_agent.txt"


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


def search(readview: dict, test_name: str, client: OpenAI, model: str) -> dict:
    """Look up lab results for a physician's request using LLM-based matching.

    The LLM is given the list of available test labels and the physician's
    request, and returns which labels (if any) correspond to the request.

    Returns:
        {"found": True,  "results": [...]}   if matched
        {"found": False, "message": "..."}   if not found
    """
    events = readview.get("events", [])
    if not events:
        return {
            "found":   False,
            "message": f"No lab results are available for this patient at this time.",
        }

    # Build a deduplicated list of available labels for the LLM to reason over
    available_labels = sorted({_label(ev) for ev in events if _label(ev)})

    system = _PROMPT.read_text(encoding="utf-8").strip()
    user_msg = (
        f"Available test labels:\n{json.dumps(available_labels)}\n\n"
        f"Physician request: \"{test_name}\"\n\n"
        "Return a JSON object: {\"matches\": [list of matching label strings]} "
        "or {\"matches\": []} if none match."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_completion_tokens=200,
    )

    try:
        matched_labels = set(json.loads(resp.choices[0].message.content).get("matches", []))
    except Exception:
        matched_labels = set()

    matches = [ev for ev in events if _label(ev) in matched_labels]
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
