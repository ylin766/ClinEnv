"""Process-level scorers — operate on dialogue logs, not outcome submissions.

Public API:
  score_info_coverage(stage_dialogue, stage_case) → dict
  score_lab_cost(stage_dialogue) → dict
  score_medication_cost(stage_dialogue, stage_case) → dict
  score_process(stage_dialogue, stage_case, conf) → dict
    Runs all enabled process scorers and returns a single merged dict.
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

    Returns merged dict with keys for each enabled scorer.
    """
    result: dict = {}

    if proc_conf.get("info_coverage", {}).get("enabled") and stage_case:
        result["info_coverage"] = score_info_coverage(stage_dialogue, stage_case)

    if proc_conf.get("lab_cost", {}).get("enabled"):
        result["lab_cost"] = score_lab_cost(stage_dialogue)

    if proc_conf.get("medication_cost", {}).get("enabled"):
        result["medication_cost"] = score_medication_cost(stage_dialogue, stage_case)

    return result


__all__ = [
    "score_info_coverage",
    "score_lab_cost",
    "score_medication_cost",
    "score_process",
]
