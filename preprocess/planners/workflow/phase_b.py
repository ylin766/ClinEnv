"""
Phase B — locate each decision in the timeline using a sliding-window tool-calling agent.
          Includes leakage scan (formerly Phase B2): expands single anchors to index_range
          when related events are found in a ±10 event window.

Input:  decisions from Phase A
Output: decisions enriched with `index` (int|None) or `index_range` ([start, end])
        Decisions matched only to pre-admission events are dropped (deleted set).
"""

import json
import re

from preprocess.loaders.ehr_loader import AdmissionRecord
from ..tools import B_TOOLS, validate_mark
from .helpers import call_llm, fmt_event_label, fmt_events, load_prompt, STEP

LOOKBACK = 10
LOOKFORWARD = 10


def phase_b(client, record: AdmissionRecord, decisions: list,
            verbose: bool = True, model: str | None = None) -> list:
    if verbose:
        print("\n========== PHASE B ==========")

    total = len(record.timeline)
    results = {d["id"]: None for d in decisions}
    deleted: set = set()

    ptr = 0
    start = 0
    last_found_end = 0
    llm_kwargs = {"model": model} if model else {}

    while ptr < len(decisions) and start < total:
        decision = decisions[ptr]
        end = min(start + STEP - 1, total - 1)
        window = record.timeline[start : end + 1]

        prompt = load_prompt("phase_b.txt").format(
            decision=json.dumps(decision),
            start=start, end=end, total=total,
            events=fmt_events(window),
        )

        messages = [{"role": "user", "content": prompt}]
        found = False
        pre_admission_blocked = False

        while True:
            resp = call_llm(client, messages, tools=B_TOOLS, **llm_kwargs)
            msg = resp.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if found:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"status": "ignored", "reason": "decision already marked"}),
                    })
                    continue

                err = validate_mark(
                    name, args,
                    current_start=start, total=total,
                    timeline=record.timeline,
                    decision_desc=decision.get("description", ""),
                )
                if err:
                    if "pre-admission" in err:
                        pre_admission_blocked = True
                        if verbose:
                            print(f'  [B] decision {decision["id"]} → pre-admission event rejected')
                    else:
                        if verbose:
                            print(f'  [B tool FAIL] {name} {args} → {err}')
                    tool_result = {"error": err}
                else:
                    if name == "mark_single":
                        results[decision["id"]] = {"index": args["index"]}
                        last_found_end = args["index"] + 1
                        found = True
                        if verbose:
                            evt = record.timeline[args["index"]]
                            print(f'  [B] decision {decision["id"]} → index {args["index"]}: {fmt_event_label(evt)}')
                    tool_result = {"status": "ok"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })

            if found or pre_admission_blocked:
                break

        if pre_admission_blocked and not found:
            deleted.add(decision["id"])
            ptr += 1
            start = last_found_end
        elif found:
            ptr += 1
            start = last_found_end
        else:
            start = end + 1
            if start >= total:
                if verbose:
                    print(f'  [B] decision {decision["id"]} not found → plan GT')
                ptr += 1
                start = last_found_end

    output = []
    for d in decisions:
        if d["id"] in deleted:
            if verbose:
                print(f'  [B] decision {d["id"]} deleted (pre-admission only)')
            continue
        r = results.get(d["id"])
        entry = dict(d)
        if r:
            entry.update(r)
        else:
            entry["index"] = None
        output.append(entry)

    if verbose:
        found_count = sum(1 for d in output if d.get("index") is not None or d.get("index_range"))
        print(f"  → {found_count}/{len(output)} decisions located ({len(deleted)} deleted)")

    return _leakage_scan(client, record, output, verbose=verbose, model=model)


def _leakage_scan(client, record: AdmissionRecord, decisions: list,
                  verbose: bool = True, model: str | None = None) -> list:
    """Bidirectional leakage scanner (formerly Phase B2)."""
    if verbose:
        print("\n---------- B leakage scan ----------")

    prompt_template = load_prompt("phase_b2.txt")
    llm_kwargs = {"model": model} if model else {}
    timeline = record.timeline
    total = len(timeline)

    output = []
    for d in decisions:
        if d.get("index_range") or d.get("index") is None:
            output.append(d)
            continue

        gt_idx = d["index"]
        back_start = max(0, gt_idx - LOOKBACK)
        fwd_end = min(total - 1, gt_idx + LOOKFORWARD)

        preceding_lines = [f"  [{i}] {fmt_event_label(timeline[i])}" for i in range(back_start, gt_idx)]
        following_lines = [f"  [{i}] {fmt_event_label(timeline[i])}" for i in range(gt_idx + 1, fwd_end + 1)]

        prompt = prompt_template.format(
            decision=json.dumps(d, ensure_ascii=False),
            gt_index=gt_idx,
            gt_event=f"[{gt_idx}] {fmt_event_label(timeline[gt_idx])}",
            preceding_events="\n".join(preceding_lines) or "  (none)",
            following_events="\n".join(following_lines) or "  (none)",
        )

        resp = call_llm(client, [{"role": "user", "content": prompt}], **llm_kwargs)
        text = resp.choices[0].message.content or ""
        m = re.search(r"\{[^}]+\}", text)
        result = {}
        if m:
            try:
                result = json.loads(m.group())
            except json.JSONDecodeError:
                pass

        if result.get("expand"):
            earliest = max(0, min(result.get("earliest_related_index", gt_idx), gt_idx))
            latest = min(total - 1, max(result.get("latest_related_index", gt_idx), gt_idx))
            if earliest < gt_idx or latest > gt_idx:
                entry = dict(d)
                del entry["index"]
                entry["index_range"] = [earliest, latest]
                if verbose:
                    print(f'  [B] decision {d["id"]}: expanded index {gt_idx} → range [{earliest},{latest}]')
                output.append(entry)
                continue

        if verbose:
            print(f'  [B] decision {d["id"]}: index {gt_idx} confirmed (no leakage)')
        output.append(d)

    if verbose:
        expanded = sum(1 for d in output if d.get("index_range"))
        single = sum(1 for d in output if d.get("index") is not None)
        print(f"  → {expanded} expanded to ranges, {single} confirmed as single index")

    return output
