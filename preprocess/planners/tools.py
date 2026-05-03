B_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mark_single",
            "description": "Mark that the current decision corresponds to a single timeline event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "The index of the matching timeline event."}
                },
                "required": ["index"]
            }
        }
    }
]

def validate_mark(name, args, current_start, total, timeline, decision_desc=""):
    if name != "mark_single":
        return f"Unknown tool: {name}"
    index = args.get("index")
    if index is None:
        return "Missing 'index'"
    # We don't necessarily constrain to current_start because phase_b2 searches bidirectionally
    # but the prompt gives events in [current_start, end].
    if index < 0 or index >= total:
        return f"Index {index} out of bounds"
    evt = timeline[index]
    if evt.get("pre_admission"):
        return "Cannot anchor to a pre-admission event."
    return None

def _drug_tokens_from_description(desc: str) -> set:
    if not desc:
        return set()
    import re
    words = re.findall(r'[a-zA-Z]+', desc.lower())
    return set(words) - {"start", "stop", "adjust", "switch", "medication", "drug", "po", "iv", "mg", "ml"}

def _event_drug_tokens(event: dict) -> set:
    src = event.get("source_table", "")
    text = ""
    if src == "hosp_pharmacy_df":
        text = event.get("medication", "")
    elif src == "hosp_prescriptions_df":
        text = event.get("drug", "")
    else:
        return set()
    return _drug_tokens_from_description(text)
