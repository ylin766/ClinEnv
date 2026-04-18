"""Decision planner.
Multi-turn OpenAI tool-calling agent that analyses one admission and produces
structured decision stages with triggers and ground-truth bindings.

The planner works through five phases, each injected as a separate user message:
  A — Understand the admission narrative (tools allowed)
  B — Anchor transitions to the event timeline (tools allowed)
  C — Build ground truth (tools allowed)
  D — Self-check: validate GT, continuity, stage count (no tools)
  E — Output final JSON (no tools)
"""

import json
import os
import re
import time
from pathlib import Path
from openai import OpenAI, RateLimitError
from preprocess.loaders.ehr_loader import AdmissionRecord
from preprocess.planners.planner_tools import TOOLS, dispatch

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts" / "planner"

def _load(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")

# Phase definitions: (label, prompt_file, tool_choice)
_PHASES = [
    ("A", "phase_a.txt", "auto"),
    ("B", "phase_b.txt", "auto"),
    ("C", "phase_c.txt", "auto"),
    ("D", "phase_d.txt", "none"),
    ("E", "phase_e.txt",  "none"),
]


# ------------------------------------------------------------------ #
# API call with rate-limit retry                                       #
# ------------------------------------------------------------------ #

def _call(client, messages: list, tool_choice: str, verbose: bool):
    attempt = 0
    while True:
        try:
            kwargs = dict(
                model="gpt-4o",
                messages=messages,
                temperature=0,
            )
            if tool_choice != "none":
                kwargs["tools"] = TOOLS
                kwargs["tool_choice"] = tool_choice
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            wait = min(2 ** attempt * 10, 300)  # 10, 20, 40, 80, 160, 300, 300...
            if verbose:
                print(f"[rate limit] waiting {wait}s (attempt {attempt + 1})...")
            time.sleep(wait)
            attempt += 1


# ------------------------------------------------------------------ #
# Planner agent loop                                                   #
# ------------------------------------------------------------------ #

def plan_decision_points(record: AdmissionRecord, verbose: bool = True) -> dict:
    """
    Run the phased planner on one AdmissionRecord.
    Returns the parsed decision-point JSON produced by the LLM.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    client = OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": _load("system.txt")},
        {
            "role": "user",
            "content": (
                f"Analyse this admission.\n"
                f"subject_id={record.subject_id}  hadm_id={record.hadm_id}"
            ),
        },
    ]

    final_content = ""

    for phase_label, prompt_file, tool_choice in _PHASES:
        if verbose:
            print(f"\n{'='*20} PHASE {phase_label} {'='*20}")

        messages.append({"role": "user", "content": _load(prompt_file)})
        phase_start = len(messages)  # mark where this phase's messages begin

        # inner loop: keep calling until no more tool calls for this phase
        while True:
            response = _call(client, messages, tool_choice, verbose)
            msg = response.choices[0].message
            messages.append(msg)

            if verbose:
                if msg.content:
                    preview = msg.content[:500] + ("..." if len(msg.content) > 500 else "")
                    print(f"[assistant] {preview}")

            if not msg.tool_calls:
                if phase_label == "E":
                    final_content = msg.content or ""
                break

            # execute tool calls and feed results back
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if verbose:
                    print(f"  [tool] {name}({json.dumps(args) if args else ''})")

                result = dispatch(record, name, args)
                result_str = json.dumps(result, ensure_ascii=False, default=str)

                if verbose:
                    preview = result_str[:200] + ("..." if len(result_str) > 200 else "")
                    print(f"         → {preview}")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result_str,
                })

        # After phase ends: truncate tool results in history to save tokens.
        # The LLM already processed them; future phases only need the reasoning.
        for i in range(phase_start, len(messages)):
            m = messages[i]
            if isinstance(m, dict) and m.get("role") == "tool":
                c = m["content"]
                if len(c) > 300:
                    messages[i] = {**m, "content": c[:300] + " …[truncated]"}
        if verbose:
            print(f"  [history] {len(messages)} messages after phase {phase_label}")

    # parse JSON from phase E output
    raw = final_content
    # try bare JSON first (phase E instructs no fences)
    try:
        parsed = json.loads(raw.strip())
        if "stages" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    # fallback: last ```json ... ``` block
    blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
    for block in reversed(blocks):
        try:
            parsed = json.loads(block)
            if "stages" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    # last resort: find outermost { containing "stages"
    for m in re.finditer(r"\{", raw):
        candidate = raw[m.start():]
        try:
            parsed = json.loads(candidate[:candidate.rfind("}") + 1])
            if "stages" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No valid stages JSON found in phase E output:\n{raw}")
