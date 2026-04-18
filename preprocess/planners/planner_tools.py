"""Planner tool definitions.
OpenAI function-call schemas and dispatcher for the decision planner agent.
Each tool maps to a method on AdmissionRecord, giving the planner read-only
access to the admission data during its exploration phase.
"""

import json
from preprocess.loaders.ehr_loader import AdmissionRecord

# ------------------------------------------------------------------ #
# Tool schemas exposed to the planner LLM                             #
# ------------------------------------------------------------------ #

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_metadata",
            "description": (
                "Return admission-level metadata: admission time, approximate "
                "discharge time, admission type/location, discharge location, "
                "patient age, gender, and date of death if recorded."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_note_section",
            "description": (
                "Extract a named section from the discharge note. "
                "Useful section names: 'Chief Complaint', "
                "'History of Present Illness', 'Past Medical History', "
                "'Brief Hospital Course', 'Discharge Medications', "
                "'Discharge Diagnosis', 'Discharge Instructions', "
                "'Pertinent Results'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_name": {
                        "type": "string",
                        "description": "Exact section header name (without colon).",
                    }
                },
                "required": ["section_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline_index",
            "description": (
                "Return a flat chronological list of all clinical timeline events. "
                "Each entry has: index (int), event_time, source_table, label. "
                "Use this to understand the full event sequence and identify "
                "natural stage boundaries as continuous index ranges."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_by_index",
            "description": (
                "Return a compact summary row for each event with index in "
                "[start, end] (inclusive). Same format as get_timeline_index. "
                "Procedure events include their icd_code in the label. "
                "Use this to survey a stage range without downloading full payloads. "
                "Call get_event_detail(index) afterwards to get the complete fields "
                "of any specific event you want to cite as GT."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end":   {"type": "integer"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_event_detail",
            "description": (
                "Return the full payload for a single timeline event by index. "
                "Use this after get_events_by_index or search_events to retrieve "
                "complete fields (e.g. icd_code, drug name) for GT citation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": (
                "Case-insensitive keyword search across all timeline event fields. "
                "Returns matching events with full payload."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_diagnoses",
            "description": (
                "Return all discharge ICD diagnosis records. "
                "These are retrospective codes not visible to the doctor model; "
                "use them to verify ground truth only."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ------------------------------------------------------------------ #
# Tool dispatcher                                                      #
# ------------------------------------------------------------------ #

def dispatch(record: AdmissionRecord, name: str, args: dict):
    """Route a tool call name + args to the corresponding AdmissionRecord method."""
    if name == "get_metadata":
        return record.get_metadata()
    if name == "get_note_section":
        return record.get_note_section(args["section_name"])
    if name == "get_timeline_index":
        return record.get_timeline_index()
    if name == "get_events_by_index":
        return record.get_events_by_index(args["start"], args["end"])
    if name == "get_event_detail":
        return record.get_event_detail(args["index"])
    if name == "search_events":
        return record.search_events(args["keyword"])
    if name == "get_diagnoses":
        return record.get_diagnoses()
    raise ValueError(f"Unknown tool: {name}")
