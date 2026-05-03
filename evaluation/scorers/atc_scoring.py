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

Fallback on RxNorm lookup failure:
  LLM normalizes the drug name (strips dose/form/salt suffixes) and retries once.
"""

from __future__ import annotations
import time
from functools import lru_cache
from pathlib import Path

import requests

_RXNORM  = "https://rxnav.nlm.nih.gov/REST"
_TIMEOUT = 8
_PROMPT_FILE = Path(__file__).parent.parent.parent / "prompts" / "scoring" / "drug_normalizer.txt"

_LEVEL_SCORES: dict[int, float] = {7: 1.0, 5: 0.8, 4: 0.6, 3: 0.3, 1: 0.1, 0: 0.0}
_BOUNDARIES = [7, 5, 4, 3, 1]

# Drugs that have a known ATC code but RxNorm returns no ATC for them.
# Keys are lowercased canonical names (after alias resolution).
_HARDCODED_ATC: dict[str, str] = {
    "ringer's lactate":          "B05BB01",
    "ringers lactate":           "B05BB01",
    "lactated ringer's":         "B05BB01",
    "compound sodium lactate":   "B05BB01",
}

# Common clinical aliases that RxNorm cannot resolve by the alias name alone.
# Keys are lowercased; values are the canonical name to use for RxNorm lookup.
_ALIASES: dict[str, str] = {
    # IV fluids / electrolytes
    "normal saline":                  "Sodium Chloride",
    "ns":                             "Sodium Chloride",
    "0.9% ns":                        "Sodium Chloride",
    "half normal saline":             "Sodium Chloride",
    "0.45% ns":                       "Sodium Chloride",
    "lactated ringers":               "Ringer's Lactate",
    "lactated ringer":                "Ringer's Lactate",
    "lactated ringer's":              "Ringer's Lactate",
    "lr":                             "Ringer's Lactate",
    "lrs":                            "Ringer's Lactate",
    "ringer's solution":              "Ringer's Lactate",
    "ringers":                        "Ringer's Lactate",
    "ivf":                            "Sodium Chloride",
    "iv fluids":                      "Sodium Chloride",
    "intravenous fluids":             "Sodium Chloride",
    "d5w":                            "Dextrose",
    "dextrose water":                 "Dextrose",
    "d5ns":                           "Dextrose",
    "d51/2ns":                        "Dextrose",
    # Opioids / pain
    "dilaudid":                       "Hydromorphone",
    "ms contin":                      "Morphine",
    "msir":                           "Morphine",
    "roxanol":                        "Morphine",
    "duragesic":                      "Fentanyl",
    "sublimaze":                      "Fentanyl",
    "percocet":                       "Oxycodone",
    "vicodin":                        "Hydrocodone",
    "ultram":                         "Tramadol",
    # Antibiotics
    "zosyn":                          "Piperacillin-Tazobactam",
    "pip-tazo":                       "Piperacillin-Tazobactam",
    "pip/tazo":                       "Piperacillin-Tazobactam",
    "flagyl":                         "Metronidazole",
    "rocephin":                       "Ceftriaxone",
    "ancef":                          "Cefazolin",
    "keflex":                         "Cephalexin",
    "levaquin":                       "Levofloxacin",
    "cipro":                          "Ciprofloxacin",
    "bactrim":                        "Sulfamethoxazole-Trimethoprim",
    "septra":                         "Sulfamethoxazole-Trimethoprim",
    "smx-tmp":                        "Sulfamethoxazole-Trimethoprim",
    "tmp-smx":                        "Sulfamethoxazole-Trimethoprim",
    "vanc":                           "Vancomycin",
    "vanco":                          "Vancomycin",
    # Cardiovascular
    "lopressor":                      "Metoprolol Tartrate",
    "toprol":                         "Metoprolol Tartrate",
    "toprol xl":                      "Metoprolol Succinate",
    "coreg":                          "Carvedilol",
    "lasix":                          "Furosemide",
    "prinivil":                       "Lisinopril",
    "zestril":                        "Lisinopril",
    "norvasc":                        "Amlodipine",
    "coumadin":                       "Warfarin",
    "plavix":                         "Clopidogrel",
    "aspirin":                        "Aspirin",
    # GI / other
    "protonix":                       "Pantoprazole",
    "prilosec":                       "Omeprazole",
    "nexium":                         "Esomeprazole",
    "zofran":                         "Ondansetron",
    "phenergan":                      "Promethazine",
    "colace":                         "Docusate",
    "miralax":                        "Polyethylene Glycol",
    "solumedrol":                     "Methylprednisolone",
    "decadron":                       "Dexamethasone",
    "ativan":                         "Lorazepam",
    "valium":                         "Diazepam",
    "versed":                         "Midazolam",
    "heparin flush":                  "Heparin",
    "insulin nph":                    "Insulin",
    "insulin regular":                "Insulin",
    "insulin glargine":               "Insulin Glargine",
    "lantus":                         "Insulin Glargine",
    "novolog":                        "Insulin Aspart",
    "humalog":                        "Insulin Lispro",
}


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
        return max(codes, key=len) if codes else None
    except Exception:
        return None


@lru_cache(maxsize=512)
def _ndcs(rxcui: str) -> list[str]:
    """Return all NDC codes for a RxNorm CUI.

    NDC (National Drug Code) is the US standard for drug identification.
    If the CUI is an ingredient, finds related Clinical Drugs first.
    Returns a list of 11-digit NDC strings (may be empty if none found).
    """
    try:
        # Try direct NDC lookup first
        r = requests.get(
            f"{_RXNORM}/rxcui/{rxcui}/ndcs.json",
            timeout=_TIMEOUT,
        )
        data = r.json()
        ndcs = data.get("ndcGroup", {}).get("ndcList", {}).get("ndc", [])
        if ndcs:
            return [str(ndc) for ndc in ndcs] if isinstance(ndcs, list) else [str(ndcs)]

        # If no NDCs, try finding related Clinical Drugs
        # Use allrelated endpoint which is more reliable
        r2 = requests.get(
            f"{_RXNORM}/rxcui/{rxcui}/allrelated.json",
            timeout=_TIMEOUT,
        )
        related_data = r2.json()
        concept_group = related_data.get("allRelatedGroup", {}).get("conceptGroup", [])

        # Look for SCD (Semantic Clinical Drug) or SBD (Semantic Branded Drug)
        # Collect NDCs from ALL related drugs so the caller can try each one.
        all_ndcs: list[str] = []
        for group in concept_group:
            tty = group.get("tty", "")
            if tty in ("SCD", "SBD"):
                for concept in group.get("conceptProperties", []):
                    related_cui = concept.get("rxcui")
                    if related_cui:
                        r3 = requests.get(
                            f"{_RXNORM}/rxcui/{related_cui}/ndcs.json",
                            timeout=_TIMEOUT,
                        )
                        raw = r3.json().get("ndcGroup", {}).get("ndcList", {}).get("ndc", [])
                        if raw:
                            batch = [str(n) for n in raw] if isinstance(raw, list) else [str(raw)]
                            all_ndcs.extend(batch)
                    if len(all_ndcs) >= 200:   # cap to avoid runaway API calls
                        break
                if len(all_ndcs) >= 200:
                    break
        return all_ndcs
    except Exception:
        return []


def _preprocess(drug_name: str) -> str:
    """Strip parenthetical brand names and normalize whitespace.

    'HYDROmorphone (Dilaudid)' → 'HYDROmorphone'
    'Piperacillin-Tazobactam (Zosyn)' → 'Piperacillin-Tazobactam'
    """
    import re as _re
    return _re.sub(r"\s*\([^)]+\)", "", drug_name).strip()


def _resolve_alias(drug_name: str) -> str:
    """Return canonical name for known clinical aliases; otherwise return input unchanged."""
    return _ALIASES.get(drug_name.strip().lower(), drug_name)


def drug_to_atc(drug_name: str) -> str | None:
    """Map a drug name to its ATC code. Returns None if the lookup fails."""
    clean    = _preprocess(drug_name)
    canonical = _resolve_alias(clean)
    # Hardcoded ATC for drugs not in RxNorm's ATC mapping
    hardcoded = _HARDCODED_ATC.get(canonical.strip().lower())
    if hardcoded:
        return hardcoded
    cui = _rxcui(canonical)
    if not cui:
        return None
    time.sleep(0.05)
    return _atc(cui)


# ------------------------------------------------------------------ #
# LLM-based drug name normalizer (fallback)                           #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=256)
def _normalize_drug_name(drug_name: str) -> str:
    """Use LLM to strip dose/form/salt suffixes; return generic name."""
    from evaluation.scorers._openai_utils import get_client, get_model
    from evaluation.scorers._usage import usage
    try:
        system_prompt = _PROMPT_FILE.read_text(encoding="utf-8").strip()
        resp = get_client().chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": drug_name},
            ],
            temperature=0,
            max_completion_tokens=20,
        )
        if hasattr(resp, "usage") and resp.usage:
            usage.track("atc_normalizer",
                        prompt_tokens=resp.usage.prompt_tokens,
                        completion_tokens=resp.usage.completion_tokens)
        normalized = resp.choices[0].message.content.strip()
        return normalized if normalized else drug_name
    except Exception:
        return drug_name


def drug_to_atc_with_fallback(drug_name: str) -> str | None:
    """Map drug name to ATC. On failure, normalize via LLM and retry once."""
    atc = drug_to_atc(drug_name)
    if atc:
        return atc
    normalized = _normalize_drug_name(drug_name)
    if normalized.lower() == drug_name.lower():
        return None
    return drug_to_atc(normalized)


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def score_medication(submitted: str, gt: str) -> float:
    """Score a submitted drug against the GT drug using ATC hierarchy.

    Returns a float in [0, 1].
    Falls back to LLM drug name normalization when RxNorm lookup fails.
    """
    if submitted.strip().lower() == gt.strip().lower():
        return 1.0

    pred_atc = drug_to_atc_with_fallback(submitted)
    gt_atc   = drug_to_atc_with_fallback(gt)

    if not pred_atc or not gt_atc:
        return 0.0

    pa, ga = pred_atc.upper(), gt_atc.upper()

    # Compare at each ATC boundary, from most to least specific.
    # cmp_len = min(boundary, actual lengths of both codes) handles cases where
    # a drug only resolves to a higher-level ATC code (fewer chars).
    # If match_len >= len(ga), the prediction covers everything the GT specifies
    # → full credit regardless of which boundary the GT stops at.
    for length in _BOUNDARIES:
        cmp_len = min(length, len(pa), len(ga))
        if cmp_len >= 1 and pa[:cmp_len] == ga[:cmp_len]:
            return 1.0 if cmp_len >= len(ga) else _LEVEL_SCORES.get(cmp_len, 0.0)

    return 0.0
