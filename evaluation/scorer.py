"""Stage and episode scoring orchestrator.

Reads episode logs produced by env/ and computes scores independently.

Outcome scoring logic:
  - procedure  : submitted text → ICD code mapping → HDF1
  - diagnosis  : submitted text → ICD code mapping → HDF1
  - medication : action match (weight 0.6) + ATC drug match (weight 0.4)
                 precision uses continuous partial credit with a floor threshold

Matching:
  Hungarian algorithm assigns each GT item to at most one submission.

Final metric: F1 score (harmonic mean of precision and recall).
  - recall    = weighted_score / total_primary_weight  (continuous, partial credit)
  - precision = sum(clipped_scores) / n_subs           (continuous, threshold-floored)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from schema import GTItem, Submission, StageResult
from evaluation.scorers.icd_scoring  import score_icd
from evaluation.scorers.atc_scoring  import score_medication
from evaluation.scorers.note_scoring import score_note_section
from env.config_loader import get_config


# ------------------------------------------------------------------ #
# GT-type → vocabulary mapping                                        #
# ------------------------------------------------------------------ #

def _detect_icd_version(code: str) -> int:
    """Infer ICD version from code format, overriding potentially wrong metadata.

    Heuristics:
      Diagnosis (CM):
        - Starts with letter other than V or E → ICD10
        - Starts with E: ICD9 E-codes are E800–E999 (digits 800–999 after E);
          anything else (e.g. E11, E119) is ICD10
        - Starts with V or is all-numeric → ICD9
      Procedure:
        - All-numeric (with optional dot) → ICD9PROC
        - Otherwise (7-char alphanumeric) → ICD10PCS
    """
    if not code:
        return 9
    c = code.strip().upper().replace(".", "")
    first = c[0]
    if first.isdigit():
        return 9
    if first == "V":
        return 9
    if first == "E":
        rest = c[1:]
        if rest.isdigit() and len(rest) >= 3 and 800 <= int(rest[:3]) <= 999:
            return 9
        return 10
    # Any other letter (A–D, F–Z except V/E) → ICD10
    return 10


def _icd_vocab(gt_item: dict) -> str:
    gt_type = gt_item.get("type", "")
    code    = gt_item.get("icd_code") or gt_item.get("value", "")
    # Use format-based detection; fall back to metadata only when code is absent
    if code:
        version = _detect_icd_version(code)
    else:
        version = gt_item.get("icd_version", 9)
    if gt_type == "procedure":
        return "ICD9PROC" if version == 9 else "ICD10PCS"
    else:
        return "ICD9CM" if version == 9 else "ICD10CM"


# ------------------------------------------------------------------ #
# Pair scoring                                                         #
# ------------------------------------------------------------------ #

def _score_medication_pair(gt_item: GTItem, submission: Submission) -> float:
    """Score a (GT medication, submission) pair.

    Action is a hard gate: if the action does not match the GT action,
    the pair scores 0 regardless of how close the drug name is.
    When action matches, the ATC-hierarchy drug score determines the final score.

    For adjust actions, direction (increase/decrease) applies a 0.5× penalty
    when both sides specify a direction and they disagree.

    If either the GT or submission lacks an action field, the gate is skipped
    and only the drug score is used (for backwards compatibility with older logs).
    """
    gt_action  = (gt_item.get("action")  or "").strip().lower()
    sub_action = (submission.get("action") or "").strip().lower()

    # Gate: action must match when both sides specify an action
    if gt_action and sub_action and gt_action != sub_action:
        return 0.0

    drug_score = score_medication(
        submission.get("value", ""),
        gt_item.get("drug_name", ""),
    )

    # Direction partial penalty for adjust
    if gt_action == "adjust":
        gt_dir  = (gt_item.get("direction")   or "").strip().lower()
        sub_dir = (submission.get("direction") or "").strip().lower()
        if gt_dir and sub_dir and gt_dir != sub_dir:
            drug_score *= 0.5

    return drug_score


def _score_pair(gt_item: GTItem, submission: Submission) -> float:
    """Score one (GT item, submission) pair. Returns float in [0, 1].

    Type mismatch always returns 0.
    """
    gt_type  = gt_item.get("type", "")
    sub_type = submission.get("type", "")

    if gt_type != sub_type:
        return 0.0

    if gt_type in ("diagnosis", "procedure"):
        icd_code = gt_item.get("icd_code")
        # Fallback: extract ICD code embedded in description as "[CODE] Title"
        if not icd_code:
            import re as _re
            m = _re.match(r"^\[([^\]]+)\]", gt_item.get("description", ""))
            if m:
                icd_code = m.group(1)
        if not icd_code:
            return 0.0
        return score_icd(
            submission.get("value", ""),
            icd_code,
            _icd_vocab({**gt_item, "icd_code": icd_code}),
        )

    if gt_type == "medication":
        return _score_medication_pair(gt_item, submission)

    return 0.0


# ------------------------------------------------------------------ #
# Matching                                                             #
# ------------------------------------------------------------------ #

def _match(gt_items: list[GTItem], submissions: list[Submission]) -> list[dict]:
    """Optimal one-to-one matching via Hungarian algorithm."""
    if not gt_items or not submissions:
        return []

    matrix = np.array(
        [[_score_pair(g, s) for s in submissions] for g in gt_items],
        dtype=float,
    )
    row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)

    return [
        {
            "gt_item":    gt_items[r],
            "submission": submissions[c],
            "score":      float(matrix[r, c]),
            "method":     "hungarian",
        }
        for r, c in zip(row_ind, col_ind)
    ]


# ------------------------------------------------------------------ #
# F1 metric                                                            #
# ------------------------------------------------------------------ #

def _f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2 * precision * recall / denom) if denom else 0.0


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def score_stage(stage_log: StageResult) -> dict:
    """Score one stage log. Returns a score dict.

    Precision is continuous (sum of clipped match scores / n_subs).
    Scores below precision_threshold are treated as 0 for precision only;
    recall still uses the full continuous partial credit.
    """
    submissions = stage_log.get("submissions", [])
    gt          = stage_log.get("gts", [])

    if not gt:
        return {"total": 0, "hits": 0.0, "recall": 0.0,
                "precision": 0.0, "f1": 0.0, "score": 0.0, "matches": []}

    matches = _match(gt, submissions)

    # ── Recall: weighted partial credit ──────────────────────────────
    eval_conf = get_config("eval")
    weights   = eval_conf.get("weights", {})
    W_PRIMARY   = weights.get("primary",   1.0)
    W_SECONDARY = weights.get("secondary", 0.8)

    gt_scores = {id(g): 0.0 for g in gt}
    for m in matches:
        gt_scores[id(m["gt_item"])] = m["score"]

    weighted_score       = 0.0
    total_primary_weight = 0.0
    for g in gt:
        is_primary = g.get("primary", True)
        weight     = W_PRIMARY if is_primary else W_SECONDARY
        weighted_score += gt_scores[id(g)] * weight
        if is_primary:
            total_primary_weight += W_PRIMARY

    if total_primary_weight == 0:
        total_primary_weight = len(gt) * W_PRIMARY
        weighted_score = sum(gt_scores.values()) * W_PRIMARY

    recall = min(1.0, weighted_score / total_primary_weight)

    # ── Precision: continuous with threshold floor ────────────────────
    med_conf  = eval_conf.get("outcome_scorers", {}).get("medication_atc", {})
    threshold = med_conf.get("precision_threshold", 0.3)

    sub_score_map: dict[int, float] = {id(s): 0.0 for s in submissions}
    for m in matches:
        sub_score_map[id(m["submission"])] = m["score"]

    clipped_sum = sum(
        s if s >= threshold else 0.0
        for s in sub_score_map.values()
    )
    precision = clipped_sum / len(submissions) if submissions else 0.0

    f1 = _f1(precision, recall)

    return {
        "total":           len(gt),
        "hits":            round(sum(gt_scores.values()), 3),
        "recall":          round(recall,    3),
        "precision":       round(precision, 3),
        "f1":              round(f1,        3),
        "score":           round(f1,        3),
        "matches":         matches,
        "all_submissions": submissions,
    }


def score_episode(episode_log: dict) -> dict:
    """Score all stages of an episode log."""
    scored_stages = []
    for stage in episode_log.get("stages", []):
        score = score_stage(stage)
        scored_stages.append({**stage, "score": score})

    return {**episode_log, "stages": scored_stages}
