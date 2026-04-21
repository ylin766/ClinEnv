"""Shared data schema — the contract between env and evaluation.

env produces StageResult objects; evaluation consumes them.
Neither module needs to know the other's internals.
"""

from __future__ import annotations
from typing import Literal, TypedDict

Mode = Literal["direct", "interactive"]


# ------------------------------------------------------------------ #
# Shared type vocabulary                                               #
# ------------------------------------------------------------------ #

DecisionType = Literal["procedure", "diagnosis", "medication", "plan"]


# ------------------------------------------------------------------ #
# Submission                                                           #
# ------------------------------------------------------------------ #

class Submission(TypedDict):
    type:      DecisionType
    value:     str   # submitted clinical text (procedure / diagnosis / drug / plan)
    reasoning: str


# ------------------------------------------------------------------ #
# Ground truth                                                         #
# ------------------------------------------------------------------ #

class GTItem(TypedDict, total=False):
    """One ground-truth item.

    Required:
        type  — decision type; drives submit tool selection and scoring.

    Value fields (present according to type):
        procedure / diagnosis : icd_code, icd_version, display, index
        medication            : drug, index
        plan                  : section, span

    Metadata (optional, not used for routing):
        source  — original data table / format (e.g. "icd_procedure", "note_section")
    """
    type:        DecisionType  # required
    # procedure / diagnosis
    icd_code:    str
    icd_version: int
    display:     str
    index:       int
    # medication
    drug:        str
    # plan
    section:     str
    span:        str
    # metadata only
    source:      str


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
