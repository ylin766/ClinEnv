"""
Phase B — locate each decision in the timeline using a sliding-window tool-calling agent.

Input:  decisions from Phase A
Output: decisions enriched with `index` (int|None) or `index_range` ([start, end])
        Decisions matched only to pre-admission events are dropped (deleted set).
"""

import json

from preprocess.loaders.ehr_loader import AdmissionRecord
from ..tools import B_TOOLS, validate_mark
from .helpers import call_llm, fmt_event_label, fmt_events, load_prompt, STEP


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

    return output
