"""Process-level scorers — operate on dialogue logs, not outcome submissions.

Public API:
  score_info_coverage(stage_dialogue, stage_case) → dict
  score_lab_cost(stage_dialogue) → dict
  score_medication_cost(stage_dialogue, stage_case) → dict
  score_process(stage_dialogue, stage_case, conf) → dict
    Runs all enabled process scorers and returns a single merged dict
    (including derived lab_efficiency metric).
"""

from __future__ import annotations

from evaluation.scorers.process_scorers.info_coverage  import score_info_coverage
from evaluation.scorers.process_scorers.lab_cost        import score_lab_cost
from evaluation.scorers.process_scorers.medication_cost import score_medication_cost


def score_process(
    stage_dialogue: dict,
    stage_case:     dict,
    proc_conf:      dict,
) -> dict:
    """Run all enabled process scorers for one stage.

    Args:
        stage_dialogue: one stage entry from dialogue.json
        stage_case:     one stage entry from case.json
        proc_conf:      process_scorers section of eval config

    Returns merged dict with keys for each enabled scorer plus lab_efficiency.
    """
    result: dict = {}

    if proc_conf.get("info_coverage", {}).get("enabled") and stage_case:
        result["info_coverage"] = score_info_coverage(stage_dialogue, stage_case)

    if proc_conf.get("lab_cost", {}).get("enabled"):
        result["lab_cost"] = score_lab_cost(stage_dialogue)

    if proc_conf.get("medication_cost", {}).get("enabled"):
        result["medication_cost"] = score_medication_cost(stage_dialogue, stage_case)

    # Derived: lab_efficiency = lab_coverage × (1 − wasted_ratio)
    #   wasted_ratio=None (no queries) → no penalty applied, but coverage=0 → 0
    if "info_coverage" in result and "lab_cost" in result:
        lab_cov   = result["info_coverage"].get("per_speaker", {}).get("lab", {}).get("coverage", 0.0)
        wasted    = result["lab_cost"].get("wasted_ratio")   # None if no queries
        waste_pen = wasted if wasted is not None else 0.0
        result["lab_efficiency"] = round(lab_cov * (1.0 - waste_pen), 3)

    return result


__all__ = [
    "score_info_coverage",
    "score_lab_cost",
    "score_medication_cost",
    "score_process",
]
