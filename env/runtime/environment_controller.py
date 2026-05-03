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

from openai import OpenAI
import anthropic
from env.llm_client import get_openai_client, get_anthropic_client, get_model_name

from schema import Mode
from env.tools.env_tools import (
    SUBMIT_TOOLS, FINALIZE_TOOL,
    get_submit_tools, get_interactive_tools, dispatch,
)

_PROMPTS_DIR        = Path(__file__).parent.parent.parent / "prompts"
_ENV_DIR            = _PROMPTS_DIR / "env"
_MUT_DIR            = _PROMPTS_DIR / "model_under_test"
_SYSTEM_PROMPT      = _ENV_DIR / "model_under_test.txt"
_INTERACTIVE_ADDON  = _ENV_DIR / "mode_interactive.txt"
_STAGE_INTERACTIVE  = _MUT_DIR / "stage_interactive.txt"
_STAGE_DIRECT       = _MUT_DIR / "stage_direct.txt"
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


def _build_stage_prompt(stage: dict, mode: Mode, hint_counts: bool = False) -> str:
    if mode == "interactive":
        label    = stage.get("label", "")
        header   = f"## {label}\n\n" if label else ""
        body     = _STAGE_INTERACTIVE.read_text(encoding="utf-8")
        if hint_counts:
            from collections import Counter
            gt_types = [g["type"] for g in stage.get("gts", []) if g.get("type")]
            counts = Counter(gt_types)
            hints = "\n\nRequired submissions for this stage:\n" + "\n".join(
                f"- {t}: {c}" for t, c in counts.items()
            )
            body += hints
        return header + body

    # direct mode — full cumulative chart up to this stage's end
    from collections import Counter
    patient_ctx = _format_patient_context(stage)
    gt_types = [g["type"] for g in stage.get("gts", []) if g.get("type")]
    required = sorted(set(gt_types))
    if hint_counts:
        counts = Counter(gt_types)
        submit_tools = ", ".join(
            f"submit_{t} ×{counts[t]}" for t in required
        ) if required else "submit_procedure / submit_diagnosis / submit_medication / submit_plan"
    else:
        submit_tools = (
            ", ".join(f"submit_{t}" for t in required)
            if required else
            "submit_procedure / submit_diagnosis / submit_medication / submit_plan"
        )
    cumulative_stage = {**stage, "events": stage.get("visible_events", stage.get("events", []))}
    return _STAGE_DIRECT.read_text(encoding="utf-8").format(
        patient_ctx  = patient_ctx,
        events       = _format_events(cumulative_stage),
        submit_tools = submit_tools,
    )


# ------------------------------------------------------------------ #
# Cross-stage context                                                  #
# ------------------------------------------------------------------ #

def _gt_summary(gt: list[dict]) -> str:
    """Format a stage's GT as confirmed clinical facts for the next stage's context."""
    lines = []
    for item in gt:
        src = item.get("source", "")
        if src == "icd_diagnosis":
            lines.append(f"- Diagnosis confirmed: {item.get('display', item.get('icd_code', ''))}")
        elif src == "icd_procedure":
            lines.append(f"- Procedure performed: {item.get('display', item.get('icd_code', ''))}")
        elif src == "medication":
            lines.append(f"- Medication decision: {item.get('drug', '')}")
        elif src == "note_section":
            lines.append(f"- Clinical decision: {item.get('span', '')}")
    return "\n".join(lines) if lines else "(no structured decisions)"


def _is_uninformative_tool_response(content: str) -> bool:
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            return False
        
        if data.get("found") is False:
            return True
        if "error" in data:
            return True
        if "message" in data and str(data["message"]).startswith("No prior"):
            return True
        
        resp = data.get("response", "").lower()
        # Only filter responses that are entirely a refusal — short responses
        # dominated by a "no data" phrase. Long responses that start with a hedge
        # ("I do not have X, but BP is 160/90...") are kept because they carry
        # useful clinical content alongside the disclaimer.
        if len(resp) < 120:
            phrases = [
                "no bedside data available",
                "information is not available",
                "no data available",
                "i do not have",
                "not have access to",
                "no record of",
            ]
            if any(p in resp for p in phrases):
                return True
    except Exception:
        pass
    return False

def _filter_messages_for_context(messages: list[dict]) -> list[dict]:
    """Return prior-stage messages with all submit/finalize activity stripped.

    Submit and finalize tool calls are removed from assistant messages entirely.
    Their corresponding tool responses are also dropped so no orphaned
    tool_call_ids remain. The GT summary is injected separately by
    _build_prior_context — the model should not see its own (possibly wrong)
    submissions in prior context. Uninformative queries are also dropped.
    """
    dropped_ids: set[str] = set()
    
    # Pass 1: find tool calls to drop
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if tc.get("function", {}).get("name") in SUBMIT_TOOLS | {FINALIZE_TOOL}:
                    dropped_ids.add(tc["id"])
        elif msg.get("role") == "tool":
            if _is_uninformative_tool_response(msg.get("content", "{}")):
                dropped_ids.add(msg.get("tool_call_id", ""))

    filtered = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            continue
        
        # Drop tool responses
        if msg.get("role") == "tool" and msg.get("tool_call_id") in dropped_ids:
            continue
            
        # Strip tool_calls from assistant messages
        if msg.get("role") == "assistant":
            remaining_tcs = [
                tc for tc in (msg.get("tool_calls") or [])
                if tc.get("id") not in dropped_ids
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


import uuid

def _build_fake_submit_messages(gt: list[dict], label: str) -> list[dict]:
    messages = []
    if not gt:
        return []
        
    for item in gt:
        t_type = item.get("type", "")
        if not t_type:
            continue
            
        tool_name = f"submit_{t_type}"
        args = {"reasoning": f"Confirmed clinical decision for {label}."}
        
        if t_type == "medication":
            args["action"]    = item.get("action", "start")
            args["drug_name"] = item.get("drug_name", item.get("drug", ""))
        elif t_type == "diagnosis":
            # description format: "[ICD] Clinical text" — strip the code prefix
            desc = item.get("description", "")
            if desc.startswith("[") and "]" in desc:
                desc = desc[desc.index("]") + 1:].strip()
            args["diagnosis"] = desc
        elif t_type == "procedure":
            args["procedure"] = item.get("description", "")
        elif t_type == "plan":
            args["plan"] = item.get("description", item.get("span", ""))
            
        tc_id = f"call_{uuid.uuid4().hex[:10]}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            }]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": json.dumps({"status": "recorded", "type": t_type})
        })
        
    tc_id = f"call_{uuid.uuid4().hex[:10]}"
    messages.append({
        "role": "assistant",
        "content": "All required decisions for this stage have been made.",
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {
                "name": "finalize_decision",
                "arguments": "{}"
            }
        }]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tc_id,
        "content": json.dumps({"status": "finalized"})
    })
    return messages


def _build_prior_context(prior_stage_logs: list[dict]) -> list[dict]:
    """Build the context injected at the start of each new stage (interactive mode).

    Includes:
      - filtered conversation history from all prior stages
      - confirmed GT decisions from each prior stage, injected as fake assistant
        tool calls so the model believes it made the correct decisions.
    """
    if not prior_stage_logs:
        return []

    context_messages: list[dict] = []
    for log in prior_stage_logs:
        # conversation history (minus actual submit/finalize turns)
        context_messages.extend(_filter_messages_for_context(log.get("messages", [])))
        # GT injected as fake model submissions
        context_messages.extend(_build_fake_submit_messages(log.get("gts", []), log.get("label", "")))

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

def _patch_gptoss_tool_calls(resp, tools: list):
    """gpt-oss via vLLM puts tool arguments in content and leaves tool_calls=[].
    Extract tool name from reasoning, arguments from content, and return a
    dict-based fake response so no OpenAI model_dump serialization is needed.
    """
    import uuid as _uuid
    if not resp or not resp.choices:
        return resp
    msg = resp.choices[0].message
    tcs = getattr(msg, "tool_calls", None) or []
    if tcs:
        return resp  # already correct format

    content   = getattr(msg, "content", "") or ""
    reasoning = getattr(msg, "reasoning", "") or getattr(msg, "reasoning_content", "") or ""

    # Try to parse content as JSON arguments
    try:
        args_dict = json.loads(content)
    except Exception:
        return resp  # not JSON, leave as-is

    # Extract tool name from reasoning (e.g. "call the ask_nurse tool")
    tool_names = [t["function"]["name"] for t in tools]
    matched = None
    for name in tool_names:
        if name in reasoning:
            matched = name
            break

    # Fallback: match by unique argument keys
    if not matched:
        arg_keys = set(args_dict.keys())
        for t in tools:
            sig_keys = set(t["function"].get("parameters", {}).get("properties", {}).keys())
            if sig_keys and arg_keys == sig_keys:
                matched = t["function"]["name"]
                break
        # finalize_decision / get_history_summary take no args
        if not matched and not arg_keys:
            for name in ["finalize_decision", "get_history_summary"]:
                if name in tool_names:
                    matched = name
                    break

    if not matched:
        return resp  # can't determine tool, leave content as-is

    # Return a dict-based message (skips model_dump entirely downstream)
    tc_id   = f"call_{_uuid.uuid4().hex[:8]}"
    msg_dict = {
        "role":       "assistant",
        "content":    None,
        "tool_calls": [{
            "id":       tc_id,
            "type":     "function",
            "function": {"name": matched, "arguments": json.dumps(args_dict)},
        }],
    }

    class FakeMsg:
        def model_dump(self): return msg_dict
    class FakeChoice:
        message = FakeMsg()
    class FakeResp:
        choices = [FakeChoice()]

    return FakeResp()


def _normalize_messages_for_vllm(messages: list) -> list:
    """Normalize messages for strict user/assistant alternation models (e.g. Gemma via vLLM).

    - tool role → user role (content wrapped with <tool_response> tags)
    - assistant messages: tool_calls converted to text, tool_calls field stripped
    - consecutive same-role messages merged (excluding system)
    """
    normalized = []
    for msg in messages:
        m = msg if isinstance(msg, dict) else msg.model_dump()
        role = m.get("role")

        if role == "tool":
            content = f"<tool_response>\n{m.get('content', '')}\n</tool_response>"
            role = "user"
            new_msg = {"role": "user", "content": content}
        elif role == "assistant":
            parts = []
            if m.get("content"):
                parts.append(m["content"])
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                parts.append(f"[Calling {fn.get('name', '')}({fn.get('arguments', '')})]")
            new_msg = {"role": "assistant", "content": "\n".join(parts) if parts else ""}
        else:
            new_msg = {"role": role, "content": m.get("content") or ""}

        # Merge consecutive same-role messages (excluding system)
        if normalized and normalized[-1]["role"] == role and role != "system":
            normalized[-1]["content"] += "\n\n" + new_msg["content"]
        else:
            normalized.append(new_msg)

    return normalized


def _call_openai(client: OpenAI, model: str, messages: list, tools: list, verbose: bool):
    from env.llm_client import chat_complete
    from env.config_loader import is_vllm_enabled
    from openai import BadRequestError

    send_messages = messages
    extra_kwargs: dict = {"tool_choice": "auto", "temperature": 0, "parallel_tool_calls": False}
    if is_vllm_enabled("runtime"):
        send_messages = _normalize_messages_for_vllm(messages)
        extra_kwargs.pop("parallel_tool_calls")  # not supported by all vLLM backends

    try:
        resp = chat_complete(
            client,
            model=model,
            messages=send_messages,
            tools=tools,
            **extra_kwargs,
        )
    except BadRequestError as e:
        # Context length exceeded (max_tokens <= 0) — force finalize
        if "max_tokens" in str(e) or "context" in str(e).lower():
            class _FinMsg:
                content = None
                tool_calls = [type("TC", (), {
                    "id": "call_ctx_limit",
                    "type": "function",
                    "function": type("F", (), {"name": "finalize_decision", "arguments": "{}"}),
                    "model_dump": lambda self: {"id": self.id, "type": self.type, "function": {"name": self.function.name, "arguments": self.function.arguments}},
                })()]
                def model_dump(self):
                    return {"role": "assistant", "content": None, "tool_calls": [tc.model_dump() for tc in self.tool_calls]}
            class _Resp:
                choices = [type("C", (), {"message": _FinMsg()})()]
            return _Resp()
        raise
    return _patch_gptoss_tool_calls(resp, tools)

def _call_anthropic(client: anthropic.Anthropic, model: str, messages: list, tools: list, verbose: bool):
    # Convert OpenAI tools to Anthropic format
    anthropic_tools = []
    for t in tools:
        fn = t["function"]
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"]
        })

    # Convert OpenAI messages to Anthropic format
    anth_messages = []
    for m in messages:
        if not isinstance(m, dict):
            m = m.model_dump()
        role = m["role"]
        if role == "system": continue
        
        content = []
        if m.get("content"):
            content.append({"type": "text", "text": m["content"]})
        
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"])
                })
        
        if role == "tool":
            # Tool responses in Anthropic are a separate 'user' message with tool_result blocks
            # But the OpenAI history might have multiple tool messages in a row.
            # We need to group them or handle them carefully.
            # Simplified: just create a user message with a tool_result
            anth_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"]
                }]
            })
            continue

        if content:
            anth_messages.append({"role": role, "content": content})

    # System prompt is separate in Anthropic
    system = next((m["content"] for m in messages if m["role"] == "system"), "")

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=anth_messages,
        system=system,
        tools=anthropic_tools,
        temperature=0
    )

    # Convert Anthropic response back to OpenAI-like object for the loop
    class FakeChoice:
        def __init__(self, message): self.message = message
    class FakeMessage:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls
        def model_dump(self): return {"role": "assistant", "content": self.content, "tool_calls": [tc.model_dump() for tc in self.tool_calls] if self.tool_calls else None}
    class FakeToolCall:
        def __init__(self, id, name, args):
            self.id = id
            self.function = type('obj', (object,), {'name': name, 'arguments': args})
        def model_dump(self): return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": self.function.arguments}}

    text = ""
    tcs = []
    for b in resp.content:
        if b.type == "text":
            text += b.text
        elif b.type == "tool_use":
            tcs.append(FakeToolCall(b.id, b.name, json.dumps(b.input)))
    
    return type('obj', (object,), {'choices': [FakeChoice(FakeMessage(text or None, tcs or None))]})

def _call(client: OpenAI | anthropic.Anthropic, messages: list, tools: list, verbose: bool):
    model = os.getenv("MUT_MODEL", _DEFAULT_MODEL)
    if isinstance(client, anthropic.Anthropic):
        return _call_anthropic(client, model, messages, tools, verbose)
    else:
        return _call_openai(client, model, messages, tools, verbose)


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
        nurse_events = rv.get("nurse", {}).get("events", [])
        if not nurse_events:
            # No bedside data — deliver the context directly rather than
            # routing through the nurse agent (who would only say "no data").
            return f"[Clinical Update] {context}" if context else None
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
    mut_client: OpenAI | anthropic.Anthropic,
    stage: dict,
    mode: Mode = "direct",
    prior_stage_logs: list[dict] | None = None,
    prior_admissions: list[dict] | None = None,
    verbose: bool = True,
    hint_counts: bool = False,
) -> dict:
    """Run one stage. Returns a stage log dict: {label, turns, submissions, gt, messages}.

    hint_counts: if True, tell the model how many submissions of each type are required
                 and decrement the count in each tool response as submissions are recorded.
    Scoring is NOT included — call evaluation.scorer.score_stage() separately.
    """
    from collections import Counter
    required_types   = {g["type"] for g in stage.get("gts", []) if g.get("type")}
    required_counts  = Counter(
        g["type"] for g in stage.get("gts", []) if g.get("type")
    )

    prior_admissions = prior_admissions or []
    available_agents = list(stage.get("available_agents", ["patient", "nurse", "lab"]))

    # Suppress agents that have no new events in the current stage window.
    # Activation criterion: at least one relevant event has index within
    # [context_range[0], context_range[1]]. Cumulative history (earlier indices)
    # is available to agents as memory, but does not justify activating them
    # if nothing new happened in this stage.
    #
    # Patient is an exception: their readview is static (CC/HPI/demographics)
    # and adds no new information after stage 0. Suppress from stage 1 onward.
    if mode == "interactive":
        rv = stage.get("readviews", {})
        stage_start, stage_end = stage.get("context_range", [0, 0])

        def _has_new_events(events: list) -> bool:
            return any(stage_start <= e.get("index", -1) <= stage_end for e in events)

        if "nurse" in available_agents:
            if not _has_new_events(rv.get("nurse", {}).get("events", [])):
                available_agents.remove("nurse")
                if verbose:
                    print("  [env] no nurse events in current stage window — nurse suppressed")

        if "lab" in available_agents:
            if not _has_new_events(rv.get("lab", {}).get("events", [])):
                available_agents.remove("lab")
                if verbose:
                    print("  [env] no lab events in current stage window — lab suppressed")

        if "patient" in available_agents and prior_stage_logs:
            available_agents.remove("patient")
            if verbose:
                print("  [env] patient readview is static — patient suppressed after stage 0")

    tools = (
        get_submit_tools(required_types) if mode == "direct"
        else get_interactive_tools(available_agents, has_history=bool(prior_admissions),
                                   required_types=required_types)
    )

    ctx = {
        "client":           client,
        "model":            os.getenv("ENV_MODEL", _DEFAULT_MODEL),
        "prior_admissions": prior_admissions,
    }

    messages: list = [{"role": "system", "content": _system_prompt(mode)}]
    if prior_stage_logs:
        if mode == "interactive":
            messages.extend(_build_prior_context(prior_stage_logs))
        else:
            # Direct mode: inject GT from each prior stage as established facts
            prior_text = "\n\n".join(
                f"[What actually happened — {log.get('label', '')}]\n{_gt_summary(log.get('gts', []))}"
                for log in prior_stage_logs
            )
            messages.append({"role": "user", "content": prior_text})
    messages.append({"role": "user", "content": _build_stage_prompt(stage, mode, hint_counts=hint_counts)})
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

        msg_obj = _call(mut_client, messages, tools, verbose)
        if not msg_obj or not msg_obj.choices:
            # vLLM returned empty response — force finalize
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": "call_empty", "type": "function",
                                             "function": {"name": "finalize_decision", "arguments": "{}"}}]})
            break
        msg     = msg_obj.choices[0].message
        
        # Convert to dict for the messages history
        msg_dict = msg if isinstance(msg, dict) else msg.model_dump()
        messages.append(msg_dict)
        
        # Use dict-style access for the rest of the loop
        msg_content = msg_dict.get("content")
        msg_tcs     = msg_dict.get("tool_calls")

        if verbose and msg_content:
            preview = msg_content[:300]
            print(f"  [model] {preview}{'...' if len(msg_content) > 300 else ''}")

        if not msg_tcs:
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

        for tc in msg_tcs:
            if isinstance(tc, dict):
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                tc_id = tc["id"]
            else:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                tc_id = tc.id

            if verbose:
                print(f"  [tool] {name}({json.dumps(args) if args else ''})")

            result = dispatch(name, args, stage, ctx=ctx)

            if verbose:
                preview = json.dumps(result, default=str)
                print(f"         → {preview[:200]}{'...' if len(preview) > 200 else ''}")

            if name in SUBMIT_TOOLS and result.get("status") == "recorded":
                duplicate = any(
                    s.get("type") == result.get("type")
                    and s.get("value") == result.get("value")
                    and s.get("action") == result.get("action")
                    for s in submissions
                )
                if not duplicate:
                    submissions.append(result)
                    if hint_counts:
                        sub_type = result.get("type", "")
                        if required_counts[sub_type] > 0:
                            required_counts[sub_type] -= 1
                        remaining = {t: c for t, c in required_counts.items() if c > 0}
                        result = {**result, "remaining": remaining if remaining else "all done"}
                elif verbose:
                    print(f"  [env] duplicate submission ignored: {result.get('type')} '{result.get('value')[:60]}'...")

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
                "tool_call_id": tc_id,
                "content":     json.dumps(result, ensure_ascii=False, default=str),
            })

        if finalized:
            break

    raw_messages = [m if isinstance(m, dict) else m.model_dump() for m in messages]
    return {
        "label":         stage.get("label", ""),
        "context_range": stage.get("context_range"),
        "turns":         turn + 1,
        "submissions":   submissions,
        "gts":           stage.get("gts", []),
        "conversation":  _build_conversation(raw_messages, start=current_stage_start),
        "messages":      raw_messages,
    }


# ------------------------------------------------------------------ #
# Episode loop                                                         #
# ------------------------------------------------------------------ #

def run_episode(
    case: dict,
    model: str | None = None,
    mode: Mode = "direct",
    verbose: bool = True,
    hint_counts: bool = False,
) -> dict:
    """Run all stages of a prepared case. Returns the raw episode log (no scores).

    To score the result, pass the returned dict to evaluation.scorer.score_episode().
    """
    # env client: always uses regular OpenAI (patient/nurse/lab agents)
    client  = get_openai_client(is_env=True)

    mut_model = get_model_name(requested_model=model)
    if "claude" in mut_model.lower():
        mut_client = get_anthropic_client()
    else:
        # mut_client uses vLLM if enabled in runtime config, else same as env client
        from env.config_loader import is_vllm_enabled
        mut_client = get_openai_client(is_env=False) if is_vllm_enabled("runtime") else client

    if model:
        os.environ["MUT_MODEL"] = mut_model

    stages           = case["stages"]
    prior_admissions = case.get("prior_admissions", [])
    stage_logs: list[dict] = []

    for i, stage in enumerate(stages, 1):
        if verbose:
            print(f"\n{'='*60}\nSTAGE {i}/{len(stages)}: {stage.get('label', '')}\n{'='*60}")
        log = run_stage(
            client,
            mut_client,
            stage,
            mode             = mode,
            prior_stage_logs = stage_logs,
            prior_admissions = prior_admissions,
            verbose          = verbose,
            hint_counts      = hint_counts,
        )
        stage_logs.append(log)

    return {
        "subject_id": case["subject_id"],
        "hadm_id":    case["hadm_id"],
        "model":      os.getenv("MUT_MODEL", _DEFAULT_MODEL),
        "mode":       mode,
        "stages":     stage_logs,
    }
