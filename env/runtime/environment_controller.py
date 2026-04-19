"""Environment controller.

Runs one complete episode: iterates through stages, runs a tool-calling loop
per stage, collects structured submissions, and returns the episode log.

Scoring is NOT done here — it is the responsibility of the evaluation/ module.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

from env.tools.env_tools import TOOLS, SUBMIT_TOOLS, FINALIZE_TOOL, dispatch

_SYSTEM_PROMPT = Path(__file__).parent.parent.parent / "prompts" / "env" / "model_under_test.txt"
_DEFAULT_MODEL  = "gpt-5.4-mini-2026-03-17"


# ------------------------------------------------------------------ #
# Prompt + formatting helpers                                          #
# ------------------------------------------------------------------ #

def _system_prompt() -> str:
    return _SYSTEM_PROMPT.read_text(encoding="utf-8")


def _format_patient_context(stage: dict) -> str:
    patient = stage.get("readviews", {}).get("patient", {})
    parts = []
    if patient.get("age") or patient.get("gender"):
        parts.append(f"**Demographics**: {patient.get('age', '?')} y/o {patient.get('gender', '?')}")
    if patient.get("chief_complaint"):
        parts.append(f"**Chief Complaint**: {patient['chief_complaint']}")
    if patient.get("hpi"):
        parts.append(f"**History of Present Illness**:\n{patient['hpi']}")
    if patient.get("past_medical_history"):
        parts.append(f"**Past Medical History**:\n{patient['past_medical_history']}")
    return "\n\n".join(parts)


def _format_events(stage: dict) -> str:
    from env.tools.env_tools import _compact_event, _all_visible_events
    events = _all_visible_events(stage)
    if not events:
        return "No clinical events recorded in this stage."
    lines = []
    for e in events:
        row = _compact_event(e)
        parts = [f"[{row.get('index')}]", row.get("time", ""), row.get("table", "")]
        if row.get("label"):   parts.append(row["label"])
        if row.get("value"):   parts.append(row["value"] + (" " + row["uom"] if row.get("uom") else ""))
        if row.get("ref_range"): parts.append(f"ref:{row['ref_range']}")
        if row.get("summary"): parts.append(row["summary"])
        if row.get("icd_code"): parts.append(f"ICD:{row['icd_code']}")
        lines.append("  " + " | ".join(str(p) for p in parts if p))
    return "\n".join(lines)


def _build_stage_prompt(stage: dict, stage_num: int, total: int) -> str:
    _GT_TO_TYPE = {"icd_procedure": "procedure", "icd_diagnosis": "diagnosis", "medication": "medication"}
    required = sorted({
        _GT_TO_TYPE[g["source"]] for g in stage.get("gt", []) if g.get("source") in _GT_TO_TYPE
    })
    required_str = (
        ", ".join(f"submit_{t}" for t in required)
        if required else
        "submit_procedure / submit_diagnosis / submit_medication"
    )
    return (
        f"## Stage {stage_num}/{total}: {stage.get('label', '')}\n\n"
        f"**Decision Required**: {stage.get('decision_required', '')}\n\n"
        f"---\n\n"
        f"{_format_patient_context(stage)}\n\n"
        f"---\n\n"
        f"## Clinical Events\n\n"
        f"{_format_events(stage)}\n\n"
        f"---\n\n"
        f"For this stage you must submit using: {required_str}\n"
        f"Then call finalize_decision."
    )


# ------------------------------------------------------------------ #
# API call with retry                                                  #
# ------------------------------------------------------------------ #

def _call(client: OpenAI, messages: list, verbose: bool):
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(
                model=os.getenv("ENV_MODEL", _DEFAULT_MODEL),
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
        except RateLimitError:
            wait = min(2 ** attempt * 10, 300)
            if verbose:
                print(f"  [rate limit] waiting {wait}s...")
            time.sleep(wait)
            attempt += 1


# ------------------------------------------------------------------ #
# Stage loop                                                           #
# ------------------------------------------------------------------ #

def run_stage(
    client: OpenAI,
    stage: dict,
    stage_num: int,
    total_stages: int,
    verbose: bool = True,
) -> dict:
    """Run one stage: tool-calling loop until finalize_decision is accepted.

    Returns a stage log dict:
        {label, index_range, turns, submissions, gt, messages}
    Note: scoring is NOT included — call evaluation.scorer.score_stage() separately.
    """
    _GT_TO_TYPE = {"icd_procedure": "procedure", "icd_diagnosis": "diagnosis", "medication": "medication"}
    required_types = {
        _GT_TO_TYPE[g["source"]]
        for g in stage.get("gt", [])
        if g.get("source") in _GT_TO_TYPE
    }

    messages: list = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user",   "content": _build_stage_prompt(stage, stage_num, total_stages)},
    ]

    submissions: list[dict] = []
    finalized = False
    max_turns = 30

    for turn in range(max_turns):
        response = _call(client, messages, verbose)
        msg = response.choices[0].message
        messages.append(msg)

        if verbose and msg.content:
            preview = msg.content[:300]
            print(f"  [model] {preview}{'...' if len(msg.content) > 300 else ''}")

        if not msg.tool_calls:
            if verbose:
                print("  [env] No tool call — prompting to submit and finalize.")
            messages.append({
                "role": "user",
                "content": (
                    "Please submit your decisions using submit_medication, submit_diagnosis, "
                    "and/or submit_procedure, then call finalize_decision."
                ),
            })
            continue

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if verbose:
                print(f"  [tool] {name}({json.dumps(args) if args else ''})")

            result = dispatch(name, args, stage)

            if verbose:
                preview = json.dumps(result, default=str)
                print(f"         → {preview[:200]}{'...' if len(preview) > 200 else ''}")

            if name in SUBMIT_TOOLS and result.get("status") == "recorded":
                submissions.append(result)

            if name == FINALIZE_TOOL:
                submitted_types = {s["type"] for s in submissions}
                missing = required_types - submitted_types
                if missing:
                    result = {
                        "status": "rejected",
                        "reason": (
                            f"Cannot finalize yet. Missing: {', '.join(sorted(missing))}. "
                            f"Please submit at least one of each required type."
                        ),
                    }
                    if verbose:
                        print(f"  [env] finalize rejected: missing {missing}")
                else:
                    finalized = True

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

        if finalized:
            break

    return {
        "label":       stage.get("label", ""),
        "index_range": stage.get("index_range"),
        "turns":       turn + 1,
        "submissions": submissions,
        "gt":          stage.get("gt", []),
        "messages":    [m if isinstance(m, dict) else m.model_dump() for m in messages],
    }


# ------------------------------------------------------------------ #
# Episode loop                                                         #
# ------------------------------------------------------------------ #

def run_episode(case: dict, model: str | None = None, verbose: bool = True) -> dict:
    """Run all stages of a prepared case. Returns the raw episode log (no scores).

    To score the result, pass the returned dict to evaluation.scorer.score_episode().
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    client  = OpenAI(api_key=api_key)
    if model:
        os.environ["ENV_MODEL"] = model

    stages = case["stages"]
    stage_logs = []

    for i, stage in enumerate(stages, 1):
        if verbose:
            print(f"\n{'='*60}\nSTAGE {i}/{len(stages)}: {stage.get('label', '')}\n{'='*60}")
        stage_logs.append(run_stage(client, stage, i, len(stages), verbose=verbose))

    return {
        "subject_id": case["subject_id"],
        "hadm_id":    case["hadm_id"],
        "model":      os.getenv("ENV_MODEL", _DEFAULT_MODEL),
        "stages":     stage_logs,
    }
