"""Environment controller.

Runs one complete episode: iterates through stages, runs a tool-calling loop
per stage, collects structured submissions, and returns the episode log.

Two modes:
  direct      — all clinical data presented upfront in the stage prompt
  interactive — model must gather data by interacting with agents

Scoring is NOT done here — it is the responsibility of the evaluation/ module.
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

from schema import Mode
from env.tools.env_tools import (
    DIRECT_TOOLS, SUBMIT_TOOLS, FINALIZE_TOOL,
    get_interactive_tools, dispatch,
)

_SYSTEM_PROMPT      = Path(__file__).parent.parent.parent / "prompts" / "env" / "model_under_test.txt"
_INTERACTIVE_ADDON  = Path(__file__).parent.parent.parent / "prompts" / "env" / "mode_interactive.txt"
_DEFAULT_MODEL      = "gpt-5.4-mini-2026-03-17"
_MAX_TURNS_INTERACT = 60


# ------------------------------------------------------------------ #
# Prompt helpers                                                       #
# ------------------------------------------------------------------ #

def _system_prompt(mode: Mode) -> str:
    base = _SYSTEM_PROMPT.read_text(encoding="utf-8")
    if mode == "interactive":
        addon = _INTERACTIVE_ADDON.read_text(encoding="utf-8")
        return f"{base}\n\n{addon}"
    return base


def _format_patient_context(stage: dict) -> str:
    patient = stage.get("readviews", {}).get("patient", {})
    parts = []
    age = patient.get("age")
    # MIMIC stores shifted year in age field when actual age is unavailable
    age_str = str(age) if age and int(age) <= 120 else "?"
    if age_str != "?" or patient.get("gender"):
        parts.append(f"**Demographics**: {age_str} y/o {patient.get('gender', '?')}")
    if patient.get("chief_complaint"):
        parts.append(f"**Chief Complaint**: {patient['chief_complaint']}")
    if patient.get("hpi"):
        parts.append(f"**History of Present Illness**:\n{patient['hpi']}")
    if patient.get("past_medical_history"):
        parts.append(f"**Past Medical History**:\n{patient['past_medical_history']}")
    return "\n\n".join(parts)


def _format_events(stage: dict) -> str:
    events = sorted(stage.get("events", []), key=lambda e: e.get("index", 0))
    if not events:
        return "No clinical events recorded in this stage."

    _compact = _compact_event
    lines = []
    for e in events:
        row = _compact(e)
        parts = [f"[{row.get('index')}]", row.get("time", ""), row.get("table", "")]
        if row.get("label"):   parts.append(row["label"])
        if row.get("value"):   parts.append(row["value"] + (" " + row["uom"] if row.get("uom") else ""))
        if row.get("ref_range"): parts.append(f"ref:{row['ref_range']}")
        if row.get("summary"): parts.append(row["summary"])
        if row.get("icd_code"): parts.append(f"ICD:{row['icd_code']}")
        lines.append("  " + " | ".join(str(p) for p in parts if p))
    return "\n".join(lines)


def _compact_event(ev: dict) -> dict:
    from typing import Any
    table = ev.get("source_table", "")
    row: dict[str, Any] = {
        "index": ev.get("index"),
        "time":  ev.get("event_time", ev.get("charttime", ev.get("storetime", ""))),
        "table": table,
    }
    if table == "hosp_pharmacy_df":
        row["label"] = ev.get("medication") or f"[{ev.get('proc_type', 'unknown')}]"
        parts = [ev.get("route", ""), ev.get("frequency", ""), ev.get("duration_interval", "")]
        row["summary"] = " | ".join(p for p in parts if p)
    elif table == "hosp_prescriptions_df":
        row["label"] = ev.get("drug", "")
        dose = f"{ev.get('dose_val_rx', '')} {ev.get('dose_unit_rx', '')}".strip()
        row["summary"] = f"{ev.get('prod_strength', '')} | {ev.get('route', '')} | {dose}".strip(" |")
    elif table == "hosp_procedures_icd_df":
        row["label"]    = ev.get("long_title_procedure", "")
        row["icd_code"] = ev.get("icd_code", "")
    elif table == "hosp_labevents_df":
        row["label"] = ev.get("label", "")
        if ev.get("value") is not None:
            row["value"] = str(ev["value"])
        if ev.get("valueuom"):
            row["uom"] = ev["valueuom"]
        if "ref_range_lower" in ev and "ref_range_upper" in ev:
            row["ref_range"] = f"{ev['ref_range_lower']}-{ev['ref_range_upper']}"
    elif table == "hosp_microbiologyevents_df":
        row["label"]   = ev.get("test_name", ev.get("org_name", ""))
        row["summary"] = ev.get("interpretation", "")
    elif table in ("ehr_chartevents_df", "ehr_datetime_events_df",
                   "ehr_outputevents_df"):
        row["label"] = ev.get("label", "")
        if ev.get("value") is not None:
            row["value"] = str(ev["value"])
        if ev.get("valueuom"):
            row["uom"] = ev["valueuom"]
    elif table in ("ehr_inputevents_df", "ehr_ingredientevents_df"):
        row["label"] = ev.get("label", "")
        if ev.get("amount") is not None:
            row["value"] = str(ev["amount"])
        if ev.get("amountuom"):
            row["uom"] = ev["amountuom"]
    elif table == "ehr_procedureevents_df":
        row["label"] = ev.get("label", "")
        if ev.get("value") is not None:
            row["value"] = str(ev["value"])
    elif table == "hosp_emar_detail_df":
        row["label"]   = ev.get("medication", ev.get("parent_field_ordername", ""))
        row["summary"] = ev.get("action_type", "")
    elif table == "radiology_note":
        row["label"]   = ev.get("label", "Radiology Report")
        row["summary"] = (ev.get("text", "")[:120] + "...") if ev.get("text") else ""
    else:
        for key in ("label", "drug", "medication", "test_name"):
            if ev.get(key):
                row["label"] = str(ev[key])
                break
        for key in ("value", "amount"):
            if ev.get(key) is not None:
                row["value"] = str(ev[key])
                break
    return {k: v for k, v in row.items() if v is not None and v != ""}


def _build_stage_prompt(stage: dict, mode: Mode) -> str:
    if mode == "interactive":
        # No patient information given upfront — the model must gather it
        # through tools. The trigger agent will open the encounter.
        return "Use the available tools to gather clinical information, then submit your decisions."

    patient_ctx = _format_patient_context(stage)
    # direct mode — include full event list
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
        f"{patient_ctx}\n\n"
        f"---\n\n"
        f"## Clinical Events\n\n"
        f"{_format_events(stage)}\n\n"
        f"---\n\n"
        f"Submit using: {required_str}, then call finalize_decision."
    )


# ------------------------------------------------------------------ #
# Cross-stage context                                                  #
# ------------------------------------------------------------------ #

def _gt_summary(gt: list[dict]) -> str:
    """Format a stage's GT as a brief clinical note for the next stage's context."""
    lines = []
    for item in gt:
        src = item.get("source", "")
        if src == "icd_diagnosis":
            lines.append(f"- Diagnosis: {item.get('display', item.get('icd_code', ''))}")
        elif src == "icd_procedure":
            lines.append(f"- Procedure: {item.get('display', item.get('icd_code', ''))}")
        elif src == "medication":
            lines.append(f"- Medication: {item.get('drug', '')}")
        elif src == "note_section":
            lines.append(f"- Decision: {item.get('span', '')}")
    return "\n".join(lines) if lines else "(no structured GT)"


def _filter_messages_for_context(messages: list[dict]) -> list[dict]:
    """Return prior-stage messages with all submit/finalize activity stripped.

    Submit and finalize tool calls are removed from assistant messages entirely.
    Their corresponding tool responses are also dropped so no orphaned
    tool_call_ids remain. The GT summary is injected separately by
    _build_prior_context — the model should not see its own (possibly wrong)
    submissions in prior context.
    """
    submit_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if tc.get("function", {}).get("name") in SUBMIT_TOOLS | {FINALIZE_TOOL}:
                    submit_ids.add(tc["id"])

    filtered = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            continue
        # Drop tool responses for submit/finalize
        if msg.get("role") == "tool" and msg.get("tool_call_id") in submit_ids:
            continue
        # Strip submit/finalize tool_calls from assistant messages
        if msg.get("role") == "assistant":
            remaining_tcs = [
                tc for tc in (msg.get("tool_calls") or [])
                if tc.get("function", {}).get("name") not in SUBMIT_TOOLS | {FINALIZE_TOOL}
            ]
            if not remaining_tcs and not msg.get("content"):
                continue  # nothing left — drop the whole message
            stripped = {**msg, "tool_calls": remaining_tcs or None}
            if stripped["tool_calls"] is None:
                stripped.pop("tool_calls", None)
            filtered.append(stripped)
        else:
            filtered.append(msg)
    return filtered


def _build_prior_context(prior_stage_logs: list[dict]) -> list[dict]:
    """Build the context injected at the start of each new stage (interactive mode).

    Includes:
      - filtered conversation history from all prior stages
      - a summary of GT decisions confirmed in each prior stage
    """
    if not prior_stage_logs:
        return []

    context_messages: list[dict] = []
    for log in prior_stage_logs:
        # conversation history (minus submit/finalize turns)
        context_messages.extend(_filter_messages_for_context(log.get("messages", [])))
        # GT summary as a system-style note
        gt_text = _gt_summary(log.get("gt", []))
        context_messages.append({
            "role":    "user",
            "content": (
                f"[Clinical decisions confirmed for the previous encounter — {log.get('label', '')}]\n"
                f"{gt_text}"
            ),
        })

    return context_messages


# ------------------------------------------------------------------ #
# Conversation builder (evaluation interface)                          #
# ------------------------------------------------------------------ #

_TOOL_SPEAKER = {
    "ask_patient": "patient",
    "ask_nurse":   "nurse",
    "order_lab":   "lab",
    "get_history_summary": "history",
    "get_history_detail":  "history",
}


def _build_conversation(messages: list[dict], start: int = 0) -> list[dict]:
    """Convert raw OpenAI messages to a clean labeled conversation.

    start: index of first message belonging to the current stage
           (messages before it are prior-context and are excluded).

    Each entry: {speaker, text, tool_call (optional)}
    Speakers: doctor | patient | nurse | lab | history | env
    Skips system messages and submit/finalize noise.
    """
    tool_id_to_name: dict[str, str] = {}
    turns = []

    for msg in messages[start:]:
        role = msg.get("role")

        if role == "system":
            continue

        if role == "assistant":
            if msg.get("content"):
                turns.append({"speaker": "doctor", "text": msg["content"]})
            for tc in msg.get("tool_calls") or []:
                fn   = tc.get("function", {})
                name = fn.get("name", "")
                tool_id_to_name[tc["id"]] = name
                if name in SUBMIT_TOOLS or name == FINALIZE_TOOL:
                    continue  # omit submit/finalize from readable log
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}
                text = args.get("question") or args.get("query") or args.get("test_name") or json.dumps(args)
                turns.append({"speaker": "doctor", "tool_call": name, "text": text})

        elif role == "tool":
            tool_name = tool_id_to_name.get(msg.get("tool_call_id", ""), "")
            if tool_name in SUBMIT_TOOLS or tool_name == FINALIZE_TOOL:
                continue
            speaker = _TOOL_SPEAKER.get(tool_name, "env")
            try:
                content = json.loads(msg.get("content", "{}"))
            except Exception:
                content = {"raw": msg.get("content")}
            text = (
                content.get("response")
                or content.get("message")
                or json.dumps({k: v for k, v in content.items() if not k.startswith("_")})
            )
            turns.append({"speaker": speaker, "tool_call": tool_name, "text": text})

        elif role == "user":
            text = msg.get("content", "")
            if text:
                turns.append({"speaker": "env", "text": text})

    return turns


# ------------------------------------------------------------------ #
# API call with retry                                                  #
# ------------------------------------------------------------------ #

def _call(client: OpenAI, messages: list, tools: list, verbose: bool):
    attempt = 0
    while True:
        try:
            return client.chat.completions.create(
                model=os.getenv("ENV_MODEL", _DEFAULT_MODEL),
                messages=messages,
                tools=tools,
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
# Trigger                                                              #
# ------------------------------------------------------------------ #

def _call_trigger(stage: dict, client: OpenAI, model: str) -> str | None:
    """Call the trigger agent to open this stage.

    The planner provides trigger.context (a one-sentence clinical fact).
    The trigger agent uses that context — plus its readview — to speak
    in their own voice, opening the encounter.

    Returns a formatted string '[Speaker] <response>', or None if the
    trigger agent is not available.
    """
    from env.agents import patient_agent, nurse_agent

    trigger  = stage.get("trigger", {})
    agent    = trigger.get("agent", "")
    context  = trigger.get("context", "")
    rv       = stage.get("readviews", {})

    if agent == "patient":
        query = (
            f"Clinical context for this encounter: {context}\n\n"
            "In character as the patient, introduce yourself and describe "
            "what brings you in today."
        ) if context else (
            "Please introduce yourself and describe what brings you in today."
        )
        response = patient_agent.answer(rv.get("patient", {}), query, client, model)
        return f"[Patient] {response}"

    if agent == "nurse":
        query = (
            f"Clinical context to communicate: {context}\n\n"
            "In character as the bedside nurse, briefly report this update "
            "to the physician, including any relevant bedside data you have."
        ) if context else (
            "Please briefly describe the patient's current clinical status "
            "and any recent developments."
        )
        response = nurse_agent.answer(rv.get("nurse", {}), query, client, model)
        return f"[Nurse] {response}"

    return None


# ------------------------------------------------------------------ #
# Stage loop                                                           #
# ------------------------------------------------------------------ #

def run_stage(
    client: OpenAI,
    stage: dict,
    mode: Mode = "direct",
    prior_stage_logs: list[dict] | None = None,
    prior_admissions: list[dict] | None = None,
    verbose: bool = True,
) -> dict:
    """Run one stage. Returns a stage log dict: {label, turns, submissions, gt, messages}.

    Scoring is NOT included — call evaluation.scorer.score_stage() separately.
    """
    _GT_TO_TYPE = {"icd_procedure": "procedure", "icd_diagnosis": "diagnosis", "medication": "medication"}
    required_types = {
        _GT_TO_TYPE[g["source"]]
        for g in stage.get("gt", [])
        if g.get("source") in _GT_TO_TYPE
    }

    prior_admissions  = prior_admissions or []
    available_agents  = stage.get("available_agents", ["patient", "nurse", "lab"])
    tools = (
        DIRECT_TOOLS if mode == "direct"
        else get_interactive_tools(available_agents, has_history=bool(prior_admissions))
    )

    ctx = {
        "client":           client,
        "model":            os.getenv("ENV_MODEL", _DEFAULT_MODEL),
        "prior_admissions": prior_admissions,
    }

    messages: list = [{"role": "system", "content": _system_prompt(mode)}]
    if mode == "interactive" and prior_stage_logs:
        messages.extend(_build_prior_context(prior_stage_logs))
    messages.append({"role": "user", "content": _build_stage_prompt(stage, mode)})
    current_stage_start = len(messages) - 1  # index of this stage's opening prompt

    # Interactive: inject trigger agent's opening message
    if mode == "interactive":
        trigger_text = _call_trigger(stage, client, os.getenv("ENV_MODEL", _DEFAULT_MODEL))
        if trigger_text:
            if verbose:
                print(f"  [trigger] {trigger_text[:200]}")
            messages.append({"role": "user", "content": trigger_text})

    submissions: list[dict] = []
    finalized  = False
    max_turns  = _MAX_TURNS_INTERACT

    for turn in range(max_turns):
        # Warn model when approaching turn limit
        if turn == max_turns - 3:
            messages.append({
                "role":    "user",
                "content": "You are approaching the turn limit. Please submit your decisions and call finalize_decision.",
            })

        response = _call(client, messages, tools, verbose)
        msg = response.choices[0].message
        messages.append(msg)

        if verbose and msg.content:
            preview = msg.content[:300]
            print(f"  [model] {preview}{'...' if len(msg.content) > 300 else ''}")

        if not msg.tool_calls:
            if verbose:
                print("  [env] No tool call — prompting to submit and finalize.")
            messages.append({
                "role":    "user",
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

            result = dispatch(name, args, stage, ctx=ctx)

            if verbose:
                preview = json.dumps(result, default=str)
                print(f"         → {preview[:200]}{'...' if len(preview) > 200 else ''}")

            if name in SUBMIT_TOOLS and result.get("status") == "recorded":
                submissions.append(result)

            if name == FINALIZE_TOOL:
                missing = required_types - {s["type"] for s in submissions}
                if missing:
                    result = {
                        "status": "rejected",
                        "reason": (
                            f"Cannot finalize yet. Missing: {', '.join(sorted(missing))}. "
                            "Please submit at least one of each required type."
                        ),
                    }
                    if verbose:
                        print(f"  [env] finalize rejected: missing {missing}")
                else:
                    finalized = True

            messages.append({
                "role":        "tool",
                "tool_call_id": tc.id,
                "content":     json.dumps(result, ensure_ascii=False, default=str),
            })

        if finalized:
            break

    raw_messages = [m if isinstance(m, dict) else m.model_dump() for m in messages]
    return {
        "label":        stage.get("label", ""),
        "index_range":  stage.get("index_range"),
        "turns":        turn + 1,
        "submissions":  submissions,
        "gt":           stage.get("gt", []),
        "conversation": _build_conversation(raw_messages, start=current_stage_start),
        "messages":     raw_messages,
    }


# ------------------------------------------------------------------ #
# Episode loop                                                         #
# ------------------------------------------------------------------ #

def run_episode(
    case: dict,
    model: str | None = None,
    mode: Mode = "direct",
    verbose: bool = True,
) -> dict:
    """Run all stages of a prepared case. Returns the raw episode log (no scores).

    To score the result, pass the returned dict to evaluation.scorer.score_episode().
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    client  = OpenAI(api_key=api_key)
    if model:
        os.environ["ENV_MODEL"] = model

    stages           = case["stages"]
    prior_admissions = case.get("prior_admissions", [])
    stage_logs: list[dict] = []

    for i, stage in enumerate(stages, 1):
        if verbose:
            print(f"\n{'='*60}\nSTAGE {i}/{len(stages)}: {stage.get('label', '')}\n{'='*60}")
        log = run_stage(
            client,
            stage,
            mode             = mode,
            prior_stage_logs = stage_logs if mode == "interactive" else None,
            prior_admissions = prior_admissions,
            verbose          = verbose,
        )
        stage_logs.append(log)

    return {
        "subject_id": case["subject_id"],
        "hadm_id":    case["hadm_id"],
        "model":      os.getenv("ENV_MODEL", _DEFAULT_MODEL),
        "mode":       mode,
        "stages":     stage_logs,
    }
