"""One-time repair script: fix medication GTs in all plan.json + case.json.

Fixes:
  1. drug_name missing  — LLM extracts drug name from GT description
  2. direction missing  — regex (then LLM fallback) extracts increase/decrease/null
                          added to ALL medication GTs with action=adjust

Updates both plan/plan.json and case/case.json in place.

Usage:
    python3 scripts/repair_gt_medication.py [--dry-run] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent.parent))
from env.llm_client import get_openai_client
from env.config_loader import get_config
from preprocess.loaders.ehr_loader import load_admission

_DATA_ROOT = str(Path(__file__).parent.parent / "mimic-ext-time-series" / "Merge" / "ehr_by_subject")

_CASES_DIR = Path(__file__).parent.parent / "data" / "cases"

# ------------------------------------------------------------------ #
# Direction: regex-first, LLM fallback                                #
# ------------------------------------------------------------------ #

_INC = re.compile(r"\b(increas|uptitrat|up.titrat|doubl|escalat|higher|bump|upward|uptitrat)\w*", re.I)
_DEC = re.compile(r"\b(decreas|reduc|down.titrat|taper|lower|halv|wean|cut.back|diminish|lessen)\w*", re.I)


def _direction_from_text(desc: str) -> str | None:
    has_inc = bool(_INC.search(desc))
    has_dec = bool(_DEC.search(desc))
    if has_inc and not has_dec:
        return "increase"
    if has_dec and not has_inc:
        return "decrease"
    return None


# ------------------------------------------------------------------ #
# LLM batch extraction                                                #
# ------------------------------------------------------------------ #

def _llm_extract(client, descriptions: list[str], model: str) -> list[dict]:
    """
    Single LLM call extracting drug_name + direction for a batch of descriptions.
    Returns list of {"drug_name": str|null, "direction": "increase"|"decrease"|null}.
    """
    numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))
    prompt = f"""You are a clinical pharmacology assistant.

For each numbered clinical decision description below, extract:
  - drug_name: the primary drug name (generic preferred; brand name acceptable if generic unclear; null if no specific drug)
  - direction: for dose-adjustment decisions only — "increase" or "decrease"; null if not a dose adjustment or direction unclear

Output a JSON array with one object per input (same order):
[
  {{"drug_name": "...", "direction": "increase"|"decrease"|null}},
  ...
]

Descriptions:
{numbered}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    # Fallback: return nulls
    return [{"drug_name": None, "direction": None}] * len(descriptions)


# ------------------------------------------------------------------ #
# Per-case repair                                                      #
# ------------------------------------------------------------------ #

def _repair_stages(stages: list, llm_results: dict[str, dict]) -> bool:
    """Apply LLM results to stages in-place. Returns True if anything changed."""
    changed = False
    for s in stages:
        for g in s.get("gts", []):
            if g.get("type") != "medication":
                continue

            desc = g.get("description", "")
            key = desc  # used to look up LLM result

            # 1. drug_name
            if not g.get("drug_name"):
                result = llm_results.get(key, {})
                dn = (result.get("drug_name") or "").strip()
                if dn and dn.lower() not in ("null", "none", "unknown"):
                    g["drug_name"] = dn
                    changed = True

            # 2. direction (all adjust GTs)
            if g.get("action") == "adjust" and not g.get("direction"):
                # Regex first
                direction = _direction_from_text(desc)
                if direction is None:
                    # Try LLM result
                    result = llm_results.get(key, {})
                    direction = result.get("direction")
                if direction in ("increase", "decrease"):
                    g["direction"] = direction
                else:
                    g["direction"] = None  # explicitly null, not missing
                changed = True

    return changed


def _emar_drug(timeline_idx: dict, event_index: int | None) -> str | None:
    """Extract drug_name from hosp_emar_detail_df event if present."""
    if event_index is None:
        return None
    ev = timeline_idx.get(event_index)
    if ev and ev.get("source_table") == "hosp_emar_detail_df":
        return ev.get("medication") or None
    return None


def repair_case(case_dir: Path, client, model: str, dry_run: bool) -> str:
    plan_file = case_dir / "plan" / "plan.json"
    case_file = case_dir / "case" / "case.json"

    if not plan_file.exists():
        return "no_plan"

    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    sid = plan["subject_id"]
    hid = plan["hadm_id"]

    # Load EHR timeline for emar_detail lookup
    try:
        rec = load_admission(sid, hid, _DATA_ROOT)
        tl_idx = {e["index"]: e for e in rec.timeline}
    except Exception:
        tl_idx = {}

    # Build an index: description -> emar drug_name (for GTs with emar anchor)
    emar_fixes: dict[str, str] = {}
    need_llm: set[str] = set()
    for s in plan.get("stages", []):
        for g in s.get("gts", []):
            if g.get("type") != "medication":
                continue
            desc = g.get("description", "")
            if not desc:
                continue
            needs_drug = not g.get("drug_name")
            needs_dir = g.get("action") == "adjust" and not g.get("direction")
            if needs_dir and _direction_from_text(desc) is None:
                need_llm.add(desc)
            if needs_drug:
                dn = _emar_drug(tl_idx, g.get("event_index"))
                if dn:
                    emar_fixes[desc] = dn
                else:
                    need_llm.add(desc)

    # LLM call (if needed)
    llm_results: dict[str, dict] = {}
    if need_llm:
        descs = list(need_llm)
        try:
            results = _llm_extract(client, descs, model)
            llm_results = {d: r for d, r in zip(descs, results)}
        except Exception as e:
            return f"llm_error:{e}"

    # Merge emar fixes into llm_results so _repair_stages handles everything uniformly
    for desc, dn in emar_fixes.items():
        if desc not in llm_results:
            llm_results[desc] = {}
        llm_results[desc]["drug_name"] = dn

    # Apply to plan
    plan_changed = _repair_stages(plan.get("stages", []), llm_results)

    # Apply to case (same GT structure)
    case_changed = False
    if case_file.exists():
        case = json.loads(case_file.read_text(encoding="utf-8"))
        case_changed = _repair_stages(case.get("stages", []), llm_results)

    if not plan_changed and not case_changed:
        return "no_change"

    if not dry_run:
        plan_file.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        if case_file.exists() and case_changed:
            case_file.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")

    return "fixed"


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run",  action="store_true", help="Show what would change, don't write")
    parser.add_argument("--workers",  type=int, default=8)
    args = parser.parse_args()

    conf  = get_config("planner")
    model = conf.get("model", "gpt-5.4-mini")
    client = get_openai_client()

    case_dirs = sorted(
        d for sid in _CASES_DIR.iterdir() if sid.is_dir()
        for d in sid.iterdir() if d.is_dir() and (d / "plan" / "plan.json").exists()
    )
    print(f"Scanning {len(case_dirs)} cases  model={model}  dry_run={args.dry_run}")

    counts = {"fixed": 0, "no_change": 0, "no_plan": 0, "llm_error": 0}

    def _one(case_dir):
        return case_dir, repair_case(case_dir, client, model, args.dry_run)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (case_dir, result) in enumerate(ex.map(_one, case_dirs), 1):
            key = result if result in counts else "llm_error"
            counts[key] += 1
            if result not in ("no_change", "no_plan"):
                print(f"  [{i}/{len(case_dirs)}] {result}  {case_dir.parent.name}/{case_dir.name}")

    print(f"\nDone: {counts}")


if __name__ == "__main__":
    main()
