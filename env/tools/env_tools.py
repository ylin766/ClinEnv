"""Environment tools for the model under test.

Two tool sets:
  DIRECT_TOOLS      — direct mode: only submit + finalize
  get_interactive_tools(has_history) — interactive mode: agents + history + submit + finalize

dispatch(name, args, stage, ctx) handles both modes.
ctx keys (interactive only): client, model, prior_admissions
"""

from __future__ import annotations
import json
from typing import Any

from env.agents import patient_agent, nurse_agent, lab_agent


# ------------------------------------------------------------------ #
# Shared: submit + finalize tool schemas                              #
# ------------------------------------------------------------------ #

_SUBMIT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_medication",
            "description": (
                "Submit one medication you would order or continue. "
                "Use the standard drug name (e.g. 'Aspirin', 'Metoprolol'). "
                "Call once per medication."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "Standard drug name."},
                    "reasoning": {"type": "string", "description": "Brief clinical reasoning."},
                },
                "required": ["drug_name", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_diagnosis",
            "description": (
                "Submit one diagnosis you have established or strongly suspect. "
                "Use ICD-compatible standard terminology "
                "(e.g. 'Acute kidney failure, unspecified'). Call once per diagnosis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string", "description": "Standard diagnosis name."},
                    "reasoning": {"type": "string", "description": "Brief clinical reasoning."},
                },
                "required": ["diagnosis", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_procedure",
            "description": (
                "Submit one procedure or intervention you would order. "
                "Use ICD-compatible standard terminology "
                "(e.g. 'Left heart cardiac catheterization'). Call once per procedure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "procedure": {"type": "string", "description": "Standard procedure name."},
                    "reasoning": {"type": "string", "description": "Brief clinical reasoning."},
                },
                "required": ["procedure", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_decision",
            "description": (
                "Finalize your decisions for this clinical encounter. "
                "Call only after all required submissions are complete."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ------------------------------------------------------------------ #
# Interactive-only tool schemas                                        #
# ------------------------------------------------------------------ #

_INTERACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_patient",
            "description": "Ask the patient a question about their symptoms, history, or concerns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Your question to the patient."},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_nurse",
            "description": (
                "Ask the bedside nurse about vitals, observations, fluid balance, "
                "or administered medications."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Your query to the nurse."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "order_lab",
            "description": (
                "Request a specific laboratory or diagnostic result by name. "
                "Returns available results or indicates the test is unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string", "description": "Name of the test to retrieve."},
                },
                "required": ["test_name"],
            },
        },
    },
]

_HISTORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_history_summary",
            "description": (
                "Retrieve a summary list of this patient's previous hospital admissions: "
                "date, chief complaint, brief hospital course, and discharge diagnosis."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history_detail",
            "description": "Retrieve the full clinical notes for a specific prior admission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based index of the admission from get_history_summary.",
                    },
                },
                "required": ["index"],
            },
        },
    },
]


# ------------------------------------------------------------------ #
# Public tool-set builders                                             #
# ------------------------------------------------------------------ #

DIRECT_TOOLS: list[dict] = _SUBMIT_TOOLS

SUBMIT_TOOLS  = {"submit_medication", "submit_diagnosis", "submit_procedure"}
FINALIZE_TOOL = "finalize_decision"

# Map agent name → tool schemas it contributes
_AGENT_TOOLS: dict[str, list[dict]] = {
    "patient": [t for t in _INTERACTION_TOOLS if t["function"]["name"] == "ask_patient"],
    "nurse":   [t for t in _INTERACTION_TOOLS if t["function"]["name"] == "ask_nurse"],
    "lab":     [t for t in _INTERACTION_TOOLS if t["function"]["name"] == "order_lab"],
    "history": _HISTORY_TOOLS,
}


def get_interactive_tools(available_agents: list[str], has_history: bool) -> list[dict]:
    """Return the interactive tool set for the given activated agents.

    available_agents: list of agent names active for this stage
                      (subset of "patient", "nurse", "lab", "history")
    has_history:      whether the case has prior admissions; gates history tools
                      even when "history" is in available_agents
    """
    tools: list[dict] = []
    for agent in available_agents:
        if agent == "history" and not has_history:
            continue
        tools.extend(_AGENT_TOOLS.get(agent, []))
    return tools + _SUBMIT_TOOLS


# ------------------------------------------------------------------ #
# Dispatch                                                             #
# ------------------------------------------------------------------ #

def dispatch(name: str, args: dict, stage: dict, ctx: dict | None = None) -> Any:
    """Route a tool call to its handler.

    ctx (optional, interactive mode):
        client            OpenAI client
        model             model name string
        prior_admissions  list of prior admission dicts (may be empty)
    """
    ctx = ctx or {}

    # ── Submit tools ─────────────────────────────────────────────────
    if name == "submit_medication":
        value = args.get("drug_name", "").strip()
        if not value:
            return {"error": "drug_name is required"}
        return {"status": "recorded", "type": "medication",
                "value": value, "reasoning": args.get("reasoning", "")}

    if name == "submit_diagnosis":
        value = args.get("diagnosis", "").strip()
        if not value:
            return {"error": "diagnosis is required"}
        return {"status": "recorded", "type": "diagnosis",
                "value": value, "reasoning": args.get("reasoning", "")}

    if name == "submit_procedure":
        value = args.get("procedure", "").strip()
        if not value:
            return {"error": "procedure is required"}
        return {"status": "recorded", "type": "procedure",
                "value": value, "reasoning": args.get("reasoning", "")}

    if name == "finalize_decision":
        return {"status": "finalized"}

    # ── Interactive: agent tools ──────────────────────────────────────
    client = ctx.get("client")
    model  = ctx.get("model", "")
    rv     = stage.get("readviews", {})

    if name == "ask_patient":
        question = args.get("question", "").strip()
        if not question:
            return {"error": "question is required"}
        return {"_speaker": "patient",
                "response": patient_agent.answer(rv.get("patient", {}), question, client, model)}

    if name == "ask_nurse":
        query = args.get("query", "").strip()
        if not query:
            return {"error": "query is required"}
        return {"_speaker": "nurse",
                "response": nurse_agent.answer(rv.get("nurse", {}), query, client, model)}

    if name == "order_lab":
        test_name = args.get("test_name", "").strip()
        if not test_name:
            return {"error": "test_name is required"}
        result = lab_agent.search(rv.get("lab", {}), test_name)
        return {"_speaker": "lab", **result}

    # ── Interactive: history tools ────────────────────────────────────
    prior = ctx.get("prior_admissions", [])

    if name == "get_history_summary":
        if not prior:
            return {"message": "No prior admissions found for this patient."}
        return {
            "count":      len(prior),
            "admissions": [
                {
                    "index":               i + 1,
                    "admittime":           p["admittime"],
                    "dischtime":           p.get("dischtime"),
                    "chief_complaint":     p.get("chief_complaint"),
                    "bhc":                 p.get("bhc"),
                    "discharge_diagnosis": p.get("discharge_diagnosis"),
                }
                for i, p in enumerate(prior)
            ],
        }

    if name == "get_history_detail":
        idx = args.get("index")
        if idx is None:
            return {"error": "index is required"}
        try:
            p = prior[int(idx) - 1]
        except (IndexError, ValueError):
            return {"error": f"No admission at index {idx}. Valid range: 1–{len(prior)}."}
        return {
            "hadm_id":   p["hadm_id"],
            "admittime": p["admittime"],
            "full_note": p.get("full_note") or "No discharge note available for this admission.",
        }

    return {"error": f"Unknown tool: {name}"}
