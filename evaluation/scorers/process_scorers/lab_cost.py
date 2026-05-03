"""Lab cost scorer.

Tracks wasted lab queries (not_found / total) and estimates USD cost
using the CMS Clinical Laboratory Fee Schedule (CLFS) via embedding + LLM rerank.
"""

from __future__ import annotations
import csv
import pickle
from pathlib import Path

import numpy as np

from env.config_loader import get_config
from env.llm_client import chat_complete
from evaluation.scorers._openai_utils import get_client, get_model
from evaluation.scorers.process_scorers._dialogue import parse_tool_messages

_CLFS_PATH    = Path(__file__).parent.parent.parent.parent / "data" / "clfs_cy2026.csv"
_PROMPT       = Path(__file__).parent.parent.parent.parent / "prompts" / "scoring" / "lab_fee_reranker.txt"
_EMB_CACHE    = Path(__file__).parent.parent.parent / "cache" / "clfs_fee_index.pkl"
_EMBED_MODEL  = "text-embedding-3-small"
_TOPK         = 15

_fee_index_cache: tuple | None = None


# ------------------------------------------------------------------ #
# CLFS data loading + embedding index                                  #
# ------------------------------------------------------------------ #

def _load_fee_schedule() -> list[dict]:
    """Load CMS CLFS → list of {cpt, fee, text} entries (lab codes only)."""
    if not _CLFS_PATH.exists():
        return []
    entries: list[dict] = []
    with open(_CLFS_PATH, newline="", encoding="utf-8") as f:
        for _ in range(4):          # skip title / copyright / PHE notice / blank
            next(f)
        reader = csv.DictReader(f)
        for row in reader:
            hcpcs = (row.get("HCPCS") or "").strip()
            if not hcpcs:
                continue
            if not (hcpcs.startswith("U") or (hcpcs.isdigit() and 80000 <= int(hcpcs) <= 89999)):
                continue
            try:
                rate = float(row.get("RATE", "0"))
            except (ValueError, TypeError):
                continue
            short     = (row.get("SHORTDESC") or "").strip()
            long_desc = (row.get("LONGDESC")  or "").strip()
            text      = f"{short}: {long_desc}" if long_desc else short
            entries.append({"cpt": hcpcs, "fee": rate, "text": text})
    return entries


def _build_index(entries: list[dict]) -> tuple:
    """Build (or load from cache) L2-normalised embedding index for CLFS entries."""
    _EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if _EMB_CACHE.exists():
        with open(_EMB_CACHE, "rb") as f:
            cached = pickle.load(f)
            if len(cached[0]) == len(entries):
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
    return payload


def _rerank(test_name: str, candidates: list[dict]) -> dict | None:
    """LLM selects the best CLFS entry; returns None if no match (selected=0)."""
    system_prompt = _PROMPT.read_text(encoding="utf-8").strip()
    lines    = "\n".join(f"{i+1}. {c['text']}" for i, c in enumerate(candidates))
    user_msg = f'Lab test: "{test_name}"\n\nCandidates:\n{lines}'
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


def _lookup_fee(test_name: str, fee_schedule: list[dict]) -> float:
    """Map a lab test name → USD via embedding retrieval + LLM rerank."""
    global _fee_index_cache
    if not test_name or not fee_schedule:
        return 0.0
    try:
        if _fee_index_cache is None:
            _fee_index_cache = _build_index(fee_schedule)
        idx_entries, embs = _fee_index_cache

        q_resp = get_client().embeddings.create(model=_EMBED_MODEL, input=[test_name])
        q      = np.array(q_resp.data[0].embedding, dtype=np.float32)
        q     /= max(float(np.linalg.norm(q)), 1e-8)

        top_idx    = list(map(int, np.argsort(embs @ q)[::-1][:_TOPK]))
        candidates = [idx_entries[i] for i in top_idx]
        selected   = _rerank(test_name, candidates)
        return selected["fee"] if selected else 0.0
    except Exception:
        return 0.0


def _lab_name_from_not_found(message: str) -> str:
    """Extract test name from 'No results available for '<name>'.' messages."""
    prefix = "No results available for '"
    if prefix in message:
        rest = message[message.index(prefix) + len(prefix):]
        return rest.split("'")[0].strip().lower()
    return ""


# ------------------------------------------------------------------ #
# Public scorer                                                        #
# ------------------------------------------------------------------ #

def score_lab_cost(stage_dialogue: dict) -> dict:
    """Score unnecessary lab query cost for one stage.

    Returns dict with:
        n_found:         int
        n_not_found:     int
        n_queries:       int
        wasted_ratio:    float | None   (None = no queries made)
        wasted_cost_usd: float
        total_cost_usd:  float
    """
    eval_conf    = get_config("eval")
    fee_path_str = eval_conf.get("process_scorers", {}).get("lab_cost", {}).get("fee_schedule", "")
    fee_schedule = _load_fee_schedule() if fee_path_str else []

    messages    = stage_dialogue.get("messages", [])
    n_found     = 0
    n_not_found = 0
    wasted_cost = 0.0
    total_cost  = 0.0

    for parsed in parse_tool_messages(messages):
        if "found" not in parsed:
            continue
        if parsed["found"]:
            n_found += 1
            for result in parsed.get("results", []):
                test_name   = result.get("test") or result.get("label") or ""
                total_cost += _lookup_fee(test_name, fee_schedule)
        else:
            n_not_found += 1
            test_name    = _lab_name_from_not_found(parsed.get("message", ""))
            cost         = _lookup_fee(test_name, fee_schedule)
            wasted_cost += cost
            total_cost  += cost

    total_queries = n_found + n_not_found
    wasted_ratio  = round(n_not_found / total_queries, 3) if total_queries > 0 else None

    return {
        "n_found":         n_found,
        "n_not_found":     n_not_found,
        "n_queries":       total_queries,
        "wasted_ratio":    wasted_ratio,
        "wasted_cost_usd": round(wasted_cost, 2),
        "total_cost_usd":  round(total_cost,  2),
    }
