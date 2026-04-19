"""ICD code scoring — H-DDx style pipeline.

Given a submitted clinical text and a ground-truth ICD code:
  1. Retrieve top-K candidates by embedding similarity (pre-built index)
  2. LLM reranker selects the best candidate
  3. Hierarchical F1 (HDF1) scores via ICD ancestor-set overlap

Supports: ICD9CM, ICD9PROC, ICD10CM, ICD10PCS (via pyhealth).
Embedding index is built once per vocabulary and cached to disk.
"""

from __future__ import annotations
import json
import pickle
import time
from pathlib import Path

import numpy as np
from pyhealth.medcode import InnerMap

from evaluation.scorers._openai_utils import get_client, get_model

# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

_CACHE_DIR   = Path(__file__).parent.parent / "cache"
_PROMPT_FILE = Path(__file__).parent.parent.parent / "prompts" / "scoring" / "icd_reranker.txt"
_EMBED_MODEL = "text-embedding-3-small"
_TOPK        = 15

_icd_vocab_cache: dict[str, InnerMap] = {}


def _vocab(name: str) -> InnerMap:
    if name not in _icd_vocab_cache:
        _icd_vocab_cache[name] = InnerMap.load(name)
    return _icd_vocab_cache[name]


def _normalize(code: str, vocab: str) -> str:
    """Convert MIMIC no-decimal codes to pyhealth decimal format.

    ICD9PROC / ICD10PCS: 2 digits before decimal  ('5325'  → '53.25')
    ICD9CM   / ICD10CM:  3 digits before decimal  ('5849'  → '584.9')
    Codes already containing '.' are returned unchanged.
    """
    if "." in code:
        return code
    code = code.strip()
    split = 2 if ("PROC" in vocab or "PCS" in vocab) else 3
    return (code[:split] + "." + code[split:]) if len(code) > split else code


# ------------------------------------------------------------------ #
# Embedding index                                                      #
# ------------------------------------------------------------------ #

def _embed(texts: list[str]) -> np.ndarray:
    resp = get_client().embeddings.create(model=_EMBED_MODEL, input=texts)
    return np.array([r.embedding for r in resp.data], dtype=np.float32)


def _load_index(vocab: str) -> tuple[list[str], np.ndarray]:
    """Load (or build and cache) the L2-normalised embedding index for a vocabulary."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _CACHE_DIR / f"{vocab}_index.pkl"

    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    print(f"  [icd_scoring] Building index for {vocab} (one-time setup)...")
    icd = _vocab(vocab)
    codes, descs = [], []
    for code in icd.graph.nodes():
        desc = icd.lookup(code)
        if desc and desc.strip():
            codes.append(code)
            descs.append(desc)

    embs = []
    for i in range(0, len(descs), 256):
        embs.append(_embed(descs[i : i + 256]))
        time.sleep(0.1)

    embeddings = np.vstack(embs)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)

    with open(cache, "wb") as f:
        pickle.dump((codes, embeddings), f)
    print(f"  [icd_scoring] Index built: {len(codes)} codes.")
    return codes, embeddings


# ------------------------------------------------------------------ #
# Retrieval + LLM reranking                                            #
# ------------------------------------------------------------------ #

def _retrieve(text: str, vocab: str) -> list[tuple[str, str, float]]:
    """Return top-K (code, description, similarity) for the given clinical text."""
    codes, embeddings = _load_index(vocab)
    icd = _vocab(vocab)

    q = _embed([text])[0]
    q /= max(np.linalg.norm(q), 1e-8)
    sims = embeddings @ q

    top_idx = np.argsort(sims)[::-1][:_TOPK]
    return [(codes[i], icd.lookup(codes[i]) or codes[i], float(sims[i])) for i in top_idx]


def _rerank(text: str, candidates: list[tuple[str, str, float]]) -> str:
    """LLM selects the best ICD code from embedding-retrieved candidates."""
    system_prompt = _PROMPT_FILE.read_text(encoding="utf-8").strip()
    candidate_lines = "\n".join(
        f"{i+1}. [{code}] {desc}" for i, (code, desc, _) in enumerate(candidates)
    )
    user_msg = f'Clinical text: "{text}"\n\nCandidates:\n{candidate_lines}'

    try:
        resp = get_client().chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_completion_tokens=20,
        )
        data = json.loads(resp.choices[0].message.content.strip())
        idx = max(0, min(int(data["selected"]) - 1, len(candidates) - 1))
        return candidates[idx][0]
    except Exception:
        return candidates[0][0]  # fallback: top-1 by embedding similarity


# ------------------------------------------------------------------ #
# Hierarchical F1                                                      #
# ------------------------------------------------------------------ #

def hierarchical_f1(pred_code: str, gt_code: str, vocab: str) -> float:
    """Compute HDF1 between a predicted and a ground-truth ICD code.

    Both codes are expanded to {code} ∪ ancestors.
    HDF1 = harmonic mean of set-precision and set-recall on the expanded sets.
    """
    icd = _vocab(vocab)

    def expand(code: str) -> set[str]:
        try:
            return {code} | set(icd.get_ancestors(code))
        except Exception:
            return {code}

    pred_set = expand(pred_code)
    gt_set   = expand(gt_code)
    inter    = pred_set & gt_set

    if not inter:
        return 0.0
    hdp = len(inter) / len(pred_set)
    hdr = len(inter) / len(gt_set)
    return (2 * hdp * hdr / (hdp + hdr)) if (hdp + hdr) else 0.0


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def score_icd(submitted_text: str, gt_code: str, vocab: str) -> float:
    """Score a submitted clinical text against a GT ICD code.

    Pipeline: text → embedding retrieval → LLM reranking → HDF1 vs GT.
    Returns a float in [0, 1].
    """
    try:
        candidates = _retrieve(submitted_text, vocab)
        if not candidates:
            return 0.0
        pred_code = _rerank(submitted_text, candidates)
        return hierarchical_f1(pred_code, _normalize(gt_code, vocab), vocab)
    except Exception:
        return 0.0
