"""Medication cost scorer.

Estimates daily drug cost per submitted medication using:
  NADAC (National Average Drug Acquisition Cost) — price per unit
  WHO ATC/DDD index                              — defined daily dose

Pipeline per drug:
  drug_name → RxNorm CUI → NDC → NADAC price/unit
                         → ATC → DDD (direct or via LLM decompose + embed + rerank)
  daily_cost = price_per_unit × DDD
"""

from __future__ import annotations
import csv
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from env.llm_client import chat_complete
from evaluation.scorers._openai_utils import get_client, get_model
from evaluation.scorers.process_scorers._dialogue import parse_tool_messages

_ROOT             = Path(__file__).parent.parent.parent.parent
_NADAC_PATH       = _ROOT / "data" / "nadac-comparison-04-29-2026.csv"
_DDD_PATH         = _ROOT / "data" / "who_atc_ddd.csv"
_DECOMPOSER_PROMPT = _ROOT / "prompts" / "scoring" / "drug_decomposer.txt"
_RERANKER_PROMPT  = _ROOT / "prompts" / "scoring" / "ddd_reranker.txt"
_EMB_CACHE        = Path(__file__).parent.parent.parent / "cache" / "ddd_index.pkl"
_EMBED_MODEL      = "text-embedding-3-small"
_TOPK             = 10

_nadac_cache:     dict[str, float] | None = None
_ddd_cache:       dict[str, tuple[float, str]] | None = None
_ddd_index_cache: tuple | None = None


# ------------------------------------------------------------------ #
# Data loading                                                         #
# ------------------------------------------------------------------ #

def _load_nadac() -> dict[str, float]:
    """Load NADAC → {ndc: price_per_unit}. Cached after first load."""
    global _nadac_cache
    if _nadac_cache is not None:
        return _nadac_cache
    nadac: dict[str, float] = {}
    if _NADAC_PATH.exists():
        with open(_NADAC_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ndc = (row.get("NDC") or "").strip()
                try:
                    price = float(row.get("New NADAC Per Unit", "0"))
                except (ValueError, TypeError):
                    continue
                if ndc and price > 0:
                    nadac[ndc] = price
    _nadac_cache = nadac
    return nadac


def _load_ddd() -> dict[str, tuple[float, str]]:
    """Load WHO DDD → {atc_code: (ddd_value, unit)}. Cached after first load."""
    global _ddd_cache
    if _ddd_cache is not None:
        return _ddd_cache
    ddd: dict[str, tuple[float, str]] = {}
    if _DDD_PATH.exists():
        with open(_DDD_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                atc     = (row.get("atc_code") or "").strip()
                ddd_str = (row.get("ddd")      or "").strip()
                uom     = (row.get("uom")      or "").strip()
                if not atc or ddd_str == "NA" or not uom:
                    continue
                try:
                    val = float(ddd_str)
                    if val > 0:
                        ddd[atc] = (val, uom)
                except (ValueError, TypeError):
                    continue
    _ddd_cache = ddd
    return ddd


def _load_ddd_entries() -> list[dict]:
    """Load all valid DDD entries as list of dicts for embedding."""
    entries = []
    if not _DDD_PATH.exists():
        return entries
    with open(_DDD_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            atc     = (row.get("atc_code") or "").strip()
            name    = (row.get("atc_name") or "").strip()
            ddd_str = (row.get("ddd")      or "").strip()
            uom     = (row.get("uom")      or "").strip()
            if not atc or not name or ddd_str == "NA" or not uom:
                continue
            try:
                val = float(ddd_str)
                if val > 0:
                    entries.append({"atc_code": atc, "atc_name": name,
                                    "ddd": val, "uom": uom,
                                    "text": f"{name} ({atc})"})
            except (ValueError, TypeError):
                continue
    return entries


# ------------------------------------------------------------------ #
# DDD embedding index                                                  #
# ------------------------------------------------------------------ #

def _build_ddd_index(entries: list[dict]) -> tuple:
    """Build (or load from cache) L2-normalised embedding index for DDD entries."""
    global _ddd_index_cache
    _EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if _EMB_CACHE.exists():
        with open(_EMB_CACHE, "rb") as f:
            cached = pickle.load(f)
            if len(cached[0]) == len(entries):
                _ddd_index_cache = cached
                return cached

    texts = [e["text"] for e in entries]
    embs  = []
    for i in range(0, len(texts), 256):
        resp = get_client().embeddings.create(model=_EMBED_MODEL, input=texts[i : i + 256])
        embs.append(np.array([r.embedding for r in resp.data], dtype=np.float32))
    embeddings = np.vstack(embs)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)

    payload = (entries, embeddings)
    with open(_EMB_CACHE, "wb") as f:
        pickle.dump(payload, f)
    _ddd_index_cache = payload
    return payload


def _get_ddd_index() -> tuple:
    global _ddd_index_cache
    if _ddd_index_cache is None:
        _ddd_index_cache = _build_ddd_index(_load_ddd_entries())
    return _ddd_index_cache


# ------------------------------------------------------------------ #
# LLM helpers: decompose + rerank                                      #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=256)
def _decompose(drug_name: str, atc_code: str | None) -> tuple[str, ...]:
    """LLM decomposes drug name into generic component names.

    Returns a tuple (for lru_cache hashability) of lowercase component strings.
    Falls back to (drug_name.lower(),) on any failure.
    """
    system_prompt = _DECOMPOSER_PROMPT.read_text(encoding="utf-8").strip()
    user_msg = f'Medication: "{drug_name}"'
    if atc_code:
        user_msg += f"\nATC code: {atc_code}"
    try:
        import json as _json
        resp = chat_complete(
            get_client(),
            model=get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_completion_tokens=100,
            response_format={"type": "json_object"},
        )
        data       = _json.loads(resp.choices[0].message.content.strip())
        components = [c.lower().strip() for c in data.get("components", []) if c]
        return tuple(components) if components else (drug_name.lower().strip(),)
    except Exception:
        return (drug_name.lower().strip(),)


def _rerank_ddd(component: str, candidates: list[dict]) -> dict | None:
    """LLM selects the best DDD entry; returns None if no match (selected=0)."""
    system_prompt = _RERANKER_PROMPT.read_text(encoding="utf-8").strip()
    lines    = "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(candidates))
    user_msg = f'Drug component: "{component}"\n\nCandidates:\n{lines}'
    try:
        import json as _json
        resp = chat_complete(
            get_client(),
            model=get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_completion_tokens=10,
        )
        data = _json.loads(resp.choices[0].message.content.strip())
        idx  = int(data.get("selected", 0))
        if idx == 0 or idx > len(candidates):
            return None
        return candidates[idx - 1]
    except Exception:
        return candidates[0] if candidates else None


# ------------------------------------------------------------------ #
# DDD lookup with decomposition + embedding + rerank                   #
# ------------------------------------------------------------------ #

def _lookup_ddd(drug_name: str, atc_code: str | None) -> tuple[float, str | None]:
    """Multi-stage DDD lookup.

    Stage 1: Direct ATC → DDD dict (fast path)
    Stage 2: LLM decomposes into components
    Stage 3: Per component → embed → top-K → LLM rerank → DDD
    Stage 4: Sum component DDDs

    Returns (total_ddd, unit) or (0.0, None).
    """
    ddd_dict = _load_ddd()

    # Stage 1: fast path
    if atc_code and atc_code in ddd_dict:
        return ddd_dict[atc_code]

    # Stage 2: decompose
    components = _decompose(drug_name, atc_code)

    # Stage 3–4: per-component embed + rerank
    try:
        idx_entries, embs = _get_ddd_index()
    except Exception:
        return (0.0, None)

    total_ddd = 0.0
    unit: str | None = None

    for component in components:
        try:
            q_resp = get_client().embeddings.create(model=_EMBED_MODEL, input=[component])
            q      = np.array(q_resp.data[0].embedding, dtype=np.float32)
            q     /= max(float(np.linalg.norm(q)), 1e-8)

            top_idx    = list(map(int, np.argsort(embs @ q)[::-1][:_TOPK]))
            candidates = [idx_entries[i] for i in top_idx]
            selected   = _rerank_ddd(component, candidates)
            if selected:
                total_ddd += selected["ddd"]
                unit       = selected["uom"]
        except Exception:
            continue

    return (total_ddd, unit) if total_ddd > 0 else (0.0, None)


# ------------------------------------------------------------------ #
# Price lookup                                                          #
# ------------------------------------------------------------------ #

def _drug_to_daily_cost(drug_name: str) -> float:
    """Map drug name → estimated daily cost (USD).

    Pipeline:
      1. Preprocess + alias resolution
      2. RxNorm CUI → NDC list → NADAC price/unit
      3. RxNorm CUI → ATC → DDD lookup (with decompose + embed + rerank fallback)
      4. daily_cost = price_per_unit × total_DDD
    """
    from evaluation.scorers.atc_scoring import (
        _rxcui, _ndcs, _atc, _preprocess, _resolve_alias,
    )
    import time

    if not drug_name:
        return 0.0

    nadac = _load_nadac()
    if not nadac:
        return 0.0

    canonical = _resolve_alias(_preprocess(drug_name))

    cui = _rxcui(canonical)
    if not cui:
        return 0.0
    time.sleep(0.05)

    # Price
    price_per_unit = 0.0
    for ndc in _ndcs(cui):
        price = nadac.get(ndc)
        if price:
            price_per_unit = price
            break
    if price_per_unit == 0.0:
        return 0.0

    # DDD
    atc_code          = _atc(cui)
    total_ddd, _unit  = _lookup_ddd(canonical, atc_code)
    if total_ddd == 0.0:
        return 0.0

    return price_per_unit * total_ddd


# ------------------------------------------------------------------ #
# Public scorer                                                        #
# ------------------------------------------------------------------ #

def score_medication_cost(stage_dialogue: dict, stage_case: dict) -> dict:
    """Score medication submission cost for one stage.

    Returns dict with:
        n_medications:    int
        total_daily_cost: float  (estimated daily cost in USD)
    """
    messages      = stage_dialogue.get("messages", [])
    n_medications = 0
    total_cost    = 0.0

    for parsed in parse_tool_messages(messages):
        if parsed.get("status") == "recorded" and parsed.get("type") == "medication":
            n_medications += 1
            drug_name      = parsed.get("value") or parsed.get("drug_name") or ""
            if drug_name:
                total_cost += _drug_to_daily_cost(drug_name)

    return {
        "n_medications":    n_medications,
        "total_daily_cost": round(total_cost, 2),
    }
