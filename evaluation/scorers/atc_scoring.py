"""ATC-based medication scoring.

Maps drug names to ATC codes via the public RxNorm API (no key required),
then scores matches using WHO ATC hierarchy partial credit.

ATC hierarchy:
  Level 1 (1 char)  Anatomical main group     e.g. C  = Cardiovascular
  Level 2 (3 chars) Therapeutic subgroup      e.g. C07 = Beta-blockers
  Level 3 (4 chars) Pharmacological subgroup  e.g. C07A
  Level 4 (5 chars) Chemical subgroup         e.g. C07AB
  Level 5 (7 chars) Chemical substance        e.g. C07AB02 = Metoprolol

Partial credit:
  Exact name match or Level-5 ATC  → 1.0
  Level-4 (same chemical subgroup) → 0.8
  Level-3 (same pharm. class)      → 0.6
  Level-2 (same therapeutic group) → 0.3
  Level-1 (same anatomical group)  → 0.1
  No match                         → 0.0
"""

from __future__ import annotations
import time
from functools import lru_cache

import requests

_RXNORM  = "https://rxnav.nlm.nih.gov/REST"
_TIMEOUT = 8

_LEVEL_SCORES: dict[int, float] = {7: 1.0, 5: 0.8, 4: 0.6, 3: 0.3, 1: 0.1, 0: 0.0}
_BOUNDARIES = [7, 5, 4, 3, 1]  # ATC prefix boundary lengths, longest first


# ------------------------------------------------------------------ #
# RxNorm / ATC lookup                                                  #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=512)
def _rxcui(drug_name: str) -> str | None:
    """Return the RxNorm CUI for a drug name (approximate match)."""
    for params, key in [
        ({"term": drug_name, "maxEntries": 5}, ("approximateGroup", "candidate")),
        ({"name": drug_name, "search": 2},     ("idGroup", "rxnormId")),
    ]:
        try:
            endpoint = "approximateTerm.json" if "term" in params else "rxcui.json"
            r = requests.get(f"{_RXNORM}/{endpoint}", params=params, timeout=_TIMEOUT)
            data = r.json()
            for k in key:
                data = data.get(k, {})
            if isinstance(data, list) and data:
                return data[0]["rxcui"] if isinstance(data[0], dict) else data[0]
        except Exception:
            pass
    return None


@lru_cache(maxsize=512)
def _atc(rxcui: str) -> str | None:
    """Return the most specific ATC code for a RxNorm CUI."""
    try:
        r = requests.get(
            f"{_RXNORM}/rxclass/class/byRxcui.json",
            params={"rxcui": rxcui, "relaSource": "ATC"},
            timeout=_TIMEOUT,
        )
        codes = [
            c["rxclassMinConceptItem"]["classId"]
            for c in r.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
            if c.get("rxclassMinConceptItem")
        ]
        return max(codes, key=len) if codes else None  # most specific = longest
    except Exception:
        return None


def drug_to_atc(drug_name: str) -> str | None:
    """Map a drug name to its ATC code. Returns None if the lookup fails."""
    cui = _rxcui(drug_name)
    if not cui:
        return None
    time.sleep(0.05)  # polite rate limiting for public API
    return _atc(cui)


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def score_medication(submitted: str, gt: str) -> float:
    """Score a submitted drug against the GT drug using ATC hierarchy.

    Returns a float in [0, 1].
    """
    # Exact name match always gives full credit (handles ATC lookup gaps)
    if submitted.strip().lower() == gt.strip().lower():
        return 1.0

    pred_atc = drug_to_atc(submitted)
    gt_atc   = drug_to_atc(gt)

    if not pred_atc or not gt_atc:
        return 0.0

    pa, ga = pred_atc.upper(), gt_atc.upper()

    # Full credit only when both codes are Level 5 (specific substance)
    if pa == ga and len(pa) >= 7:
        return 1.0

    for length in _BOUNDARIES:
        if len(pa) >= length and len(ga) >= length and pa[:length] == ga[:length]:
            return _LEVEL_SCORES[length]

    return 0.0
