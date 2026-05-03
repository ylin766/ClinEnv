"""
Post-processing: merge stages whose context window is too small.

Rules
-----
- Any stage with fewer than `min_events` context events is a candidate.
- Prefer merging into the NEXT neighbour (smaller or equal size preferred);
  keep the next stage's GT.
- Merge into PREV only when there is no next neighbour;
  keep whichever GT remains strictly after the new combined context_end.
- Always merge stages with exactly 1 context event.
"""


def merge_small_stages(stages: list, min_events: int = 10, verbose: bool = True) -> list:
    if len(stages) <= 1:
        return stages

    def ctx_size(s):
        r = s["context_range"]
        return r[1] - r[0] + 1

    def gt_index(s):
        g = s["gt"]
        if "event_index" in g:
            return g["event_index"]
        if "event_range" in g:
            return g["event_range"][1]
        return None

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(stages):
            s = stages[i]
            if ctx_size(s) < min_events:
                prev_size = ctx_size(stages[i - 1]) if i > 0 else float("inf")
                next_size = ctx_size(stages[i + 1]) if i < len(stages) - 1 else float("inf")

                # Merge into NEXT (preferred)
                if i < len(stages) - 1 and next_size <= prev_size:
                    nxt = stages[i + 1]
                    new_start = s["context_range"][0]
                    new_end = nxt["context_range"][1]
                    merged = {"context_range": [new_start, new_end], "gt": nxt["gt"]}
                    if verbose:
                        print(f"  [merge] stage {i} (ctx={ctx_size(s)}) → next; "
                              f"context=[{new_start},{new_end}]")
                    stages = stages[:i] + [merged] + stages[i + 2:]
                    changed = True
                    continue

                # Merge into PREV
                elif i > 0:
                    prv = stages[i - 1]
                    new_start = prv["context_range"][0]
                    new_end = s["context_range"][1]
                    prv_gt_idx = gt_index(prv)
                    if prv_gt_idx is None or prv_gt_idx > new_end:
                        surviving_gt, kept = prv["gt"], "prev"
                    else:
                        surviving_gt, kept = s["gt"], "small"
                    merged = {"context_range": [new_start, new_end], "gt": surviving_gt}
                    if verbose:
                        print(f"  [merge] stage {i} (ctx={ctx_size(s)}) → prev; "
                              f"context=[{new_start},{new_end}] gt from {kept}")
                    stages = stages[:i - 1] + [merged] + stages[i + 1:]
                    changed = True
                    i = max(0, i - 1)
                    continue

            i += 1

    return stages
