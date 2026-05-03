"""
Phase B2 — bidirectional leakage scanner.

After Phase B marks each decision with a single anchor `index`, this phase
looks both backward and forward from the anchor (up to 10 events each way)
to detect related records. If found, the single index is expanded into an
`index_range` to prevent answer leakage into the context window.

Input:  decisions from Phase B (each with `index` or already discarded)
Output: same decisions, with anchors potentially expanded to `index_range`
"""

import json
import re

from preprocess.loaders.ehr_loader import AdmissionRecord
from .helpers import call_llm, fmt_event_label, load_prompt


LOOKBACK = 10
LOOKFORWARD = 10


def phase_b2(client, record: AdmissionRecord, decisions: list,
             verbose: bool = True, model: str | None = None) -> list:
    if verbose:
        print("\n========== PHASE B2 (leakage scan) ==========")

    prompt_template = load_prompt("phase_b2.txt")
    llm_kwargs = {"model": model} if model else {}
    timeline = record.timeline
    total = len(timeline)

    output = []
    for d in decisions:
        # Skip decisions that already have a range or no index
        if d.get("index_range") or d.get("index") is None:
            output.append(d)
            continue

        gt_idx = d["index"]

        # Backward window
        back_start = max(0, gt_idx - LOOKBACK)
        back_end = gt_idx - 1

        # Forward window
        fwd_start = gt_idx + 1
        fwd_end = min(total - 1, gt_idx + LOOKFORWARD)

        # Format preceding events
        preceding_lines = []
        if back_start <= back_end:
            for i in range(back_start, back_end + 1):
                preceding_lines.append(f"  [{i}] {fmt_event_label(timeline[i])}")
        preceding_str = "\n".join(preceding_lines) if preceding_lines else "  (none)"

        # Format following events
        following_lines = []
        if fwd_start <= fwd_end:
            for i in range(fwd_start, fwd_end + 1):
                following_lines.append(f"  [{i}] {fmt_event_label(timeline[i])}")
        following_str = "\n".join(following_lines) if following_lines else "  (none)"

        gt_event = timeline[gt_idx]

        prompt = prompt_template.format(
            decision=json.dumps(d, ensure_ascii=False),
            gt_index=gt_idx,
            gt_event=f"[{gt_idx}] {fmt_event_label(gt_event)}",
            preceding_events=preceding_str,
            following_events=following_str,
        )

        resp = call_llm(client, [{"role": "user", "content": prompt}], **llm_kwargs)
        text = resp.choices[0].message.content or ""

        # Parse JSON from response
        m = re.search(r"\{[^}]+\}", text)
        if m:
            try:
                result = json.loads(m.group())
            except json.JSONDecodeError:
                result = {"expand": False}
        else:
            result = {"expand": False}

        if result.get("expand"):
            earliest = result.get("earliest_related_index", gt_idx)
            latest = result.get("latest_related_index", gt_idx)
            # Sanity checks
            earliest = max(0, min(earliest, gt_idx))
            latest = min(total - 1, max(latest, gt_idx))

            if earliest < gt_idx or latest > gt_idx:
                entry = dict(d)
                del entry["index"]
                entry["index_range"] = [earliest, latest]
                if verbose:
                    print(f'  [B2] decision {d["id"]}: expanded index {gt_idx} → range [{earliest},{latest}]')
                    for ri in range(earliest, latest + 1):
                        print(f'       [{ri}] {fmt_event_label(timeline[ri])}')
                output.append(entry)
                continue

        if verbose:
            print(f'  [B2] decision {d["id"]}: index {gt_idx} confirmed (no leakage)')
        output.append(d)

    if verbose:
        expanded = sum(1 for d in output if d.get("index_range"))
        single = sum(1 for d in output if d.get("index") is not None)
        print(f"  → {expanded} expanded to ranges, {single} confirmed as single index")

    return output
