"""Environment tools for the model under test.

Tool sets:
  get_submit_tools(required_types)   — direct mode: subset of submit tools + finalize
  get_interactive_tools(...)         — interactive mode: agents + submit + finalize

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
                "Submit a medication decision. Use the appropriate action:\n"
                "- 'start': order a drug for the first time\n"
                "- 'stop': discontinue a drug the patient is currently taking\n"
                "- 'adjust': change the dose or frequency of a current drug\n"
                "- 'switch': replace one drug with another (e.g. IV to oral, or different agent)\n"
                "Do NOT use submit_plan for medication changes — use this tool for all drug decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop", "adjust", "switch"],
                        "description": "The medication action being taken.",
                    },
                    "drug_name": {"type": "string", "description": "Standard drug name."},
                    "direction": {
                        "type": "string",
                        "enum": ["increase", "decrease"],
                        "description": "For 'adjust' only: whether the dose/frequency is being increased or decreased.",
                    },
                    "reasoning": {"type": "string", "description": "Brief clinical reasoning."},
                },
                "required": ["action", "drug_name", "reasoning"],
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
            "name": "submit_plan",
            "description": (
                "Submit ONE clinical management decision as a concise statement. "
                "Each distinct decision requires a separate call — do not combine multiple decisions into one submission. "
                "Examples of one decision: 'hold anticoagulation given active bleeding', "
                "'transfuse 1 unit PRBC for symptomatic anemia', 'consult urology for hematuria management'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string", "description": "One clinical management decision as a concise statement."},
                    "reasoning": {"type": "string", "description": "Brief clinical reasoning for this specific decision."},
                },
                "required": ["plan", "reasoning"],
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
            "description": "Request a specific laboratory or diagnostic result by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string", "description": "Name of the lab test or component to retrieve."},
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

SUBMIT_TOOLS  = {"submit_medication", "submit_diagnosis", "submit_procedure", "submit_plan"}
FINALIZE_TOOL = "finalize_decision"

# Map agent name → tool schemas it contributes
_AGENT_TOOLS: dict[str, list[dict]] = {
    "patient": [t for t in _INTERACTION_TOOLS if t["function"]["name"] == "ask_patient"],
    "nurse":   [t for t in _INTERACTION_TOOLS if t["function"]["name"] == "ask_nurse"],
    "lab":     [t for t in _INTERACTION_TOOLS if t["function"]["name"] == "order_lab"],
    "history": _HISTORY_TOOLS,
}

_SUBMIT_BY_TYPE: dict[str, dict] = {
    t["function"]["name"].replace("submit_", ""): t
    for t in _SUBMIT_TOOLS
    if t["function"]["name"].startswith("submit_")
}
_FINALIZE_SCHEMA = next(t for t in _SUBMIT_TOOLS if t["function"]["name"] == "finalize_decision")


def get_submit_tools(required_types: set[str]) -> list[dict]:
    """Return only the submit tools that match the stage's GT types, plus finalize.

    required_types: set of strings from {"medication", "diagnosis", "procedure", "plan"}.
    If empty, all submit tools are included as a fallback.
    """
    if not required_types:
        return _SUBMIT_TOOLS  # fallback — include all
    tools = [_SUBMIT_BY_TYPE[t] for t in required_types if t in _SUBMIT_BY_TYPE]
    tools.append(_FINALIZE_SCHEMA)
    return tools


def get_interactive_tools(
    available_agents: list[str],
    has_history: bool,
    required_types: set[str] | None = None,
) -> list[dict]:
    """Return the interactive tool set for the given activated agents.

    available_agents: list of agent names active for this stage
                      (subset of "patient", "nurse", "lab", "history")
    has_history:      whether the case has prior admissions; gates history tools
                      even when "history" is in available_agents
    required_types:   GT types for this stage; restricts available submit tools.
                      Pass None to include all submit tools.
    """
    tools: list[dict] = []
    for agent in available_agents:
        if agent == "history" and not has_history:
            continue
        tools.extend(_AGENT_TOOLS.get(agent, []))
    return tools + get_submit_tools(required_types or set())


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
        action = args.get("action", "start").strip()
        value = args.get("drug_name", "").strip()
        if not value:
            return {"error": "drug_name is required"}
        result: dict = {"status": "recorded", "type": "medication",
                        "action": action, "value": value,
                        "reasoning": args.get("reasoning", "")}
        direction = (args.get("direction") or "").strip().lower()
        if direction in ("increase", "decrease"):
            result["direction"] = direction
        return result

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

    if name == "submit_plan":
        value = args.get("plan", "").strip()
        if not value:
            return {"error": "plan is required"}
        return {"status": "recorded", "type": "plan",
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
        result = lab_agent.search(rv.get("lab", {}), test_name, client, model)
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
