"""Shared data schema — the contract between env and evaluation.

env produces StageResult objects; evaluation consumes them.
Neither module needs to know the other's internals.
"""

from __future__ import annotations
from typing import Literal, TypedDict

Mode = Literal["direct", "interactive"]


# ------------------------------------------------------------------ #
# Submission                                                           #
# ------------------------------------------------------------------ #

SubmissionType = Literal["diagnosis", "procedure", "medication"]


class Submission(TypedDict):
    type:      SubmissionType
    value:     str   # the submitted clinical text (diagnosis / procedure / drug name)
    reasoning: str


# ------------------------------------------------------------------ #
# Ground truth                                                         #
# ------------------------------------------------------------------ #

GTSource = Literal["icd_diagnosis", "icd_procedure", "medication", "note_section"]


class _GTItemBase(TypedDict):
    source: GTSource


class GTItem(_GTItemBase, total=False):
    """One ground-truth item. Fields present depend on source:

    icd_diagnosis / icd_procedure : icd_code, icd_version
    medication                    : drug
    note_section                  : span
    """
    icd_code:    str
    icd_version: int
    drug:        str
    span:        str


# ------------------------------------------------------------------ #
# Stage result — the env → evaluation interface                       #
# ------------------------------------------------------------------ #

class StageResult(TypedDict):
    """Produced by env.run_stage(); consumed by evaluation.score_stage().

    Additional logging fields (turns, messages, index_range) may be
    present in the full episode log but are not part of this interface.
    """
    label:       str
    gt:          list[GTItem]
    submissions: list[Submission]
