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
    ("D", "phase_d.txt", "auto"),
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
                model="gpt-5.4-mini-2026-03-17",
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

        # Phase D: track whether validate_gt_indices returned PASS
        _d_indices_passed  = False
        _d_novelty_passed  = False
        _d_correction_count = 0
        _D_MAX_CORRECTIONS  = 6  # max forced re-prompts before giving up

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
                # Phase D: if validation has not passed, force another iteration
                if phase_label == "D" and not _d_indices_passed and _d_correction_count < _D_MAX_CORRECTIONS:
                    _d_correction_count += 1
                    correction = (
                        "validate_gt_indices has NOT returned PASS. You must fix the violations before proceeding.\n\n"
                        "For stages with EMPTY GT: find GT items from the Brief Hospital Course. "
                        "If all timeline procedures are pre_admission=true and no significant medications exist, "
                        "use plan GT — pick one attending-level decision per stage from the BHC "
                        "(e.g. antibiotic choice, anticoagulation start, key management pivot). "
                        "One plan GT item per distinct decision.\n\n"
                        "For stages with GT index <= stage end: set that stage's end to (GT index - 1). "
                        "If this creates a gap before the next stage, insert a new stage covering the remaining events.\n\n"
                        "Call validate_gt_indices immediately with the corrected stages. Do not output any text first."
                    )
                    messages.append({"role": "user", "content": correction})
                    if verbose:
                        print(f"  [phase D] forcing correction #{_d_correction_count}")
                    continue  # loop again without breaking
                # Phase D: if novelty check has not passed, force fix
                content = msg.content or ""
                if (phase_label == "D" and _d_indices_passed and not _d_novelty_passed
                        and _d_correction_count < _D_MAX_CORRECTIONS):
                    _d_correction_count += 1
                    correction = (
                        "validate_gt_novelty returned FAIL. One or more medication/procedure GT items already appear "
                        "in the timeline before the stage ends — they are pharmacy refreshes, not new decisions.\n\n"
                        "For each violation: remove the medication GT with the repeated index, and replace it with "
                        "a plan GT from the Brief Hospital Course that describes the physician decision to start or "
                        "escalate that treatment (e.g. 'given several 500cc boluses of NS' for a fluid decision).\n\n"
                        "Alternatively, find a different qualifying medication that appears for the FIRST time after "
                        "the stage's end index.\n\n"
                        "Call validate_gt_indices then validate_gt_novelty again with the corrected stages."
                    )
                    messages.append({"role": "user", "content": correction})
                    if verbose:
                        print(f"  [phase D] forcing novelty fix #{_d_correction_count}")
                    continue
                # Phase D: if mechanical checks passed but manual checks not all PASS, force correction
                all_manual_pass = (
                    "Check A: PASS" in content
                    and "Check B: PASS" in content
                    and "Check C: PASS" in content
                )
                if (phase_label == "D" and _d_indices_passed and _d_novelty_passed
                        and _d_correction_count < _D_MAX_CORRECTIONS
                        and not all_manual_pass):
                    _d_correction_count += 1
                    # Determine which checks failed
                    failed = []
                    if "Check A: PASS" not in content:
                        failed.append("A")
                    if "Check B: PASS" not in content:
                        failed.append("B")
                    if "Check C: PASS" not in content:
                        failed.append("C")
                    correction = (
                        f"Manual check(s) {', '.join(failed)} did not PASS. "
                        f"You must fix the issues and re-run the mechanical tools before writing APPROVED.\n\n"
                        "Fix all issues identified in the failing checks, then:\n"
                        "1. Call validate_gt_indices with the corrected stages\n"
                        "2. Call validate_gt_novelty with the corrected stages\n"
                        "3. Re-state all three checks (A, B, C) — each must say PASS\n"
                        "4. Only then write the APPROVED PLAN\n\n"
                        "Key rules for fixing:\n"
                        "- Check A diagnosis filter: remove all chronic comorbidities not actively managed this admission; "
                        "keep only primary admission diagnoses and conditions that triggered a specific decision\n"
                        "- Check B duplicate: if the same decision appears as both a medication GT (with index) AND a plan GT, "
                        "keep the medication GT and remove the plan GT; if the same plan text appears in two stages, "
                        "keep it in the stage where the information that triggered it first appeared\n"
                        "- Check C trigger: rewrite any trigger that names the GT drug/procedure/diagnosis"
                    )
                    messages.append({"role": "user", "content": correction})
                    if verbose:
                        print(f"  [phase D] forcing manual check fix #{_d_correction_count} (failed: {failed})")
                    continue
                if phase_label == "E":
                    final_content = msg.content or ""
                    # Quick structural check on Phase E JSON output
                    try:
                        _parsed = json.loads(final_content)
                        _e_violations = []
                        for _s in _parsed.get("stages", []):
                            _ir = _s.get("index_range", [0, 0])
                            try:
                                _start, _end = int(_ir[0]), int(_ir[1])
                            except (TypeError, ValueError, IndexError):
                                _e_violations.append(
                                    f"stage '{_s.get('label')}': malformed index_range {_ir}"
                                )
                                continue
                            if _start > _end:
                                _e_violations.append(
                                    f"stage '{_s.get('label')}': start {_start} > end {_end} — invalid range"
                                )
                            for _g in _s.get("gt", []):
                                _idx = _g.get("index")
                                if _idx is not None:
                                    try:
                                        if int(_idx) <= _end:
                                            _e_violations.append(
                                                f"stage '{_s.get('label')}': GT index {_idx} <= end {_end}"
                                            )
                                    except (TypeError, ValueError):
                                        pass
                        if _e_violations:
                            fix_msg = (
                                "The JSON you output has structural violations: "
                                + "; ".join(_e_violations)
                                + ". Rules: (1) stage start must be <= end; "
                                "(2) every GT item with an index must have index > stage end. "
                                "Fix by adjusting stage boundaries — copy the APPROVED ranges from Phase D exactly. "
                                "Output the corrected JSON only."
                            )
                            messages.append({"role": "user", "content": fix_msg})
                            if verbose:
                                print(f"  [phase E] structural fix: {_e_violations}")
                            continue  # re-run Phase E loop
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass  # let parse failure surface naturally later
                break

            # execute tool calls and feed results back
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if verbose:
                    print(f"  [tool] {name}({json.dumps(args) if args else ''})")

                result = dispatch(record, name, args)
                result_str = json.dumps(result, ensure_ascii=False, default=str)

                # Track Phase D validation status
                if phase_label == "D":
                    if name == "validate_gt_indices":
                        _d_indices_passed = (result.get("status") == "PASS")
                    elif name == "validate_gt_novelty":
                        _d_novelty_passed = (result.get("status") == "PASS")

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
            pass
        else:
            parsed = None
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        # fallback: last ```json ... ``` block
        blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw)
        for block in reversed(blocks):
            try:
                p = json.loads(block)
                if "stages" in p:
                    parsed = p
                    break
            except json.JSONDecodeError:
                continue

    if parsed is None:
        # last resort: find outermost { containing "stages"
        for m in re.finditer(r"\{", raw):
            candidate = raw[m.start():]
            try:
                p = json.loads(candidate[:candidate.rfind("}") + 1])
                if "stages" in p:
                    parsed = p
                    break
            except json.JSONDecodeError:
                continue

    if parsed is None:
        raise ValueError(f"No valid stages JSON found in phase E output:\n{raw}")

    return parsed
