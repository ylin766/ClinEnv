"""Dialogue log parsing utilities shared across process scorers."""

from __future__ import annotations
import json


def parse_tool_messages(messages: list[dict]) -> list[dict]:
    """Return list of parsed tool result dicts from a stage's message list."""
    results = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        try:
            results.append(json.loads(content))
        except Exception:
            pass
    return results


def tool_name_from_result(parsed: dict) -> str:
    """Infer which tool was called from the result structure."""
    if "_speaker" in parsed:
        return f"ask_{parsed['_speaker']}"
    if "found" in parsed and "results" in parsed:
        return "query_lab"
    if "count" in parsed and "admissions" in parsed:
        return "get_prior_admissions"
    if "hadm_id" in parsed and "full_note" in parsed:
        return "get_prior_note"
    if parsed.get("status") in ("recorded", "finalized"):
        return "submit"
    return "unknown"


def info_tool_results(messages: list[dict]) -> list[dict]:
    """Return only information-gathering tool results (not submits/finalize)."""
    return [
        r for r in parse_tool_messages(messages)
        if tool_name_from_result(r) not in ("submit", "unknown")
        and r.get("status") not in ("recorded", "finalized")
    ]


def speaker_responses(messages: list[dict], speaker: str) -> list[str]:
    """Collect all text responses from a given speaker in the dialogue."""
    responses = []
    for parsed in parse_tool_messages(messages):
        if parsed.get("_speaker") != speaker:
            continue
        resp = parsed.get("response", "")
        if resp:
            responses.append(resp)
        # Lab: structured results (order_lab returns found+results, no response field)
        if speaker == "lab" and parsed.get("found") and parsed.get("results"):
            responses.append(json.dumps(parsed["results"]))
    return responses
