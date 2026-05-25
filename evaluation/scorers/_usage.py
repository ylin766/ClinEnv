"""Lightweight per-eval-run token usage tracker.

Usage pattern:
    from evaluation.scorers._usage import usage
    usage.reset()                  # call once before an eval run starts
    usage.track("icd_reranker", prompt_tokens=120, completion_tokens=8)
    report = usage.snapshot()      # returns structured dict
"""

from __future__ import annotations
from threading import Lock


class _UsageTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._data: dict[str, dict] = {}

    def reset(self) -> None:
        with self._lock:
            self._data = {}

    def track(self, scorer: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        with self._lock:
            if scorer not in self._data:
                self._data[scorer] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            entry = self._data[scorer]
            entry["calls"]             += 1
            entry["prompt_tokens"]     += prompt_tokens
            entry["completion_tokens"] += completion_tokens

    def snapshot(self) -> dict:
        with self._lock:
            total_p = total_c = 0
            by_scorer: dict[str, dict] = {}
            for name, d in self._data.items():
                by_scorer[name] = dict(d)
                total_p += d["prompt_tokens"]
                total_c += d["completion_tokens"]
            return {
                "total_tokens":       total_p + total_c,
                "prompt_tokens":      total_p,
                "completion_tokens":  total_c,
                "by_scorer":          by_scorer,
            }


class _ProgressTracker:
    """Global counter for LLM calls across all parallel workers.

    Usage:
        progress.reset(total=108)          # once, before work starts
        msg = progress.tick("10076617 stage_b lab")   # thread-safe; returns formatted line
    """

    def __init__(self) -> None:
        self._lock  = Lock()
        self._done  = 0
        self._total = 0

    def reset(self, total: int = 0) -> None:
        with self._lock:
            self._done  = 0
            self._total = total

    def tick(self, label: str = "") -> str:
        with self._lock:
            self._done += 1
            done, total = self._done, self._total
        pct = done / total * 100 if total else 0.0
        bar_width = 20
        filled = int(bar_width * done / total) if total else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        return f"[{bar}] {done}/{total} ({pct:.1f}%)  {label}"


# Module-level singletons — imported everywhere as:
#   from evaluation.scorers._usage import usage
#   from evaluation.scorers._usage import progress
usage    = _UsageTracker()
progress = _ProgressTracker()
