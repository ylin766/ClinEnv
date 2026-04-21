"""Stage and episode scoring orchestrator.

Reads episode logs produced by env/ and computes scores independently.

Scoring logic:
  - icd_procedure : submitted procedure text → ICD code mapping → HDF1
  - icd_diagnosis : submitted diagnosis text → ICD code mapping → HDF1
  - medication    : submitted drug name → ATC hierarchy partial credit
  - note_section  : LLM binary judge across all submissions

Matching:
  Hungarian algorithm assigns each GT item to at most one submission
  (and vice versa), maximising total score. note_section items are
  handled separately (pool of all submissions, not one-to-one).

Final metric: F1 score (harmonic mean of precision and recall).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from schema import GTItem, Submission, StageResult
from evaluation.scorers.icd_scoring  import score_icd
from evaluation.scorers.atc_scoring  import score_medication
from evaluation.scorers.note_scoring import score_note_section


# ------------------------------------------------------------------ #
# GT-type → vocabulary mapping                                        #
# ------------------------------------------------------------------ #

def _icd_vocab(gt_item: dict) -> str:
    """Return the pyhealth vocabulary name for a GT item."""
    gt_type = gt_item.get("type", "")
    version = gt_item.get("icd_version", 9)
    if gt_type == "procedure":
        return "ICD9PROC" if version == 9 else "ICD10PCS"
    else:
        return "ICD9CM" if version == 9 else "ICD10CM"


# ------------------------------------------------------------------ #
# Pair scoring                                                         #
# ------------------------------------------------------------------ #

def _score_pair(gt_item: GTItem, submission: Submission) -> float:
    """Score one (GT item, submission) pair. Returns float in [0, 1].

    Driven by gt type. Type mismatch returns 0.
    """
    gt_type  = gt_item.get("type", "")
    sub_type = submission.get("type", "")
    value    = submission.get("value", "")

    if gt_type != sub_type:
        return 0.0

    if gt_type in ("procedure", "diagnosis"):
        return score_icd(value, gt_item["icd_code"], _icd_vocab(gt_item))

    if gt_type == "medication":
        return score_medication(value, gt_item.get("drug", ""))

    if gt_type == "plan":
        result = score_note_section(gt_item.get("span", ""), [submission])
        return result["score"]

    return 0.0


# ------------------------------------------------------------------ #
# Matching                                                             #
# ------------------------------------------------------------------ #

def _match(gt_items: list[GTItem], submissions: list[Submission]) -> list[dict]:
    """Optimal one-to-one matching via Hungarian algorithm.

    All GT types use the same pair-scoring + Hungarian matching.
    Returns a list of match dicts: {gt_item, submission, score, method}.
    """
    if not gt_items or not submissions:
        return []

    matrix = np.array(
        [[_score_pair(g, s) for s in submissions] for g in gt_items],
        dtype=float,
    )
    row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)

    matches = []
    for r, c in zip(row_ind, col_ind):
        gt_type = gt_items[r].get("type", "")
        matches.append({
            "gt_item":    gt_items[r],
            "submission": submissions[c],
            "score":      float(matrix[r, c]),
            "method":     "llm_binary" if gt_type == "plan" else "hungarian",
        })
    return matches


# ------------------------------------------------------------------ #
# F2 metric                                                            #
# ------------------------------------------------------------------ #

def _f1(precision: float, recall: float) -> float:
    """F1 score — harmonic mean of precision and recall."""
    denom = precision + recall
    return (2 * precision * recall / denom) if denom else 0.0


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def score_stage(stage_log: StageResult) -> dict:
    """Score one stage log. Returns a score dict added to the stage_log.

    Input:  stage_log with keys 'submissions' and 'gt'
    Output: score dict with recall, precision, F2, and per-match details
    """
    submissions = stage_log.get("submissions", [])
    gt          = stage_log.get("gt", [])

    if not gt:
        return {"total": 0, "hits": 0.0, "recall": 0.0,
                "precision": 0.0, "f1": 0.0, "score": 0.0, "matches": []}

    matches = _match(gt, submissions)

    # Recall: average match score across all GT items
    gt_scores = {id(g): 0.0 for g in gt}
    for m in matches:
        gt_scores[id(m["gt_item"])] = m["score"]
    recall = sum(gt_scores.values()) / len(gt)

    # Precision: fraction of submissions that matched (score > 0)
    matched_sub_ids = {id(m["submission"]) for m in matches if m["score"] > 0}
    precision = len(matched_sub_ids) / len(submissions) if submissions else 1.0

    f1 = _f1(precision, recall)

    return {
        "total":      len(gt),
        "hits":       round(sum(gt_scores.values()), 3),
        "recall":     round(recall, 3),
        "precision":  round(precision, 3),
        "f1":         round(f1, 3),
        "score":      round(f1, 3),
        "matches":    matches,
        "all_submissions": submissions,
    }


def score_episode(episode_log: dict) -> dict:
    """Score all stages of an episode log. Returns enriched episode dict.

    Adds 'score' to each stage and computes overall weighted F2.
    Does NOT modify the original dict in place — returns a new one.
    """
    scored_stages = []
    for stage in episode_log.get("stages", []):
        score = score_stage(stage)
        scored_stages.append({**stage, "score": score})

    total_gt   = sum(s["score"]["total"] for s in scored_stages)
    total_hits = sum(s["score"]["hits"]  for s in scored_stages)
    total_subs = sum(len(s.get("submissions", [])) for s in scored_stages)
    matched    = sum(
        len([m for m in s["score"]["matches"] if m["score"] > 0])
        for s in scored_stages
    )

    overall_recall    = total_hits / total_gt if total_gt else 0.0
    overall_precision = matched / total_subs  if total_subs else 1.0
    overall_f1        = _f1(overall_precision, overall_recall)

    return {
        **episode_log,
        "stages":        scored_stages,
        "overall_score": round(overall_f1, 3),
        "overall_hits":  round(total_hits, 3),
        "overall_total": total_gt,
    }
