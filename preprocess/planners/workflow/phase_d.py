"""
Phase D — convert each stage's single raw `gt` into a typed `gts` list.

For each stage:
  - If the GT has anchor events: one LLM call with decision + all anchor events
    → produces typed GT items (medication/procedure/diagnosis/plan + action)
  - If no anchor (plan-only): fallback single LLM classify call
  - Two dedup passes remove redundant items
"""

import json
import re

from .helpers import call_llm, fmt_event_label, load_prompt

VALID_TYPES = {"medication", "procedure", "diagnosis", "plan"}
VALID_ACTIONS = {"start", "stop", "switch", "adjust"}


# ------------------------------------------------------------------ #
# JSON parsing helper                                                  #
# ------------------------------------------------------------------ #

def _parse_json(raw: str) -> dict:
    """
    Parse LLM JSON output robustly.
    Strips markdown code fences, then tries strict parse;
    falls back to greedy { … } extraction.
    """
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{[\s\S]*\}", stripped)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, ValueError):
                pass
        return {}


# ------------------------------------------------------------------ #
# LLM helper calls                                                     #
# ------------------------------------------------------------------ #

def _llm_action(client, desc: str, model_kwargs: dict) -> str:
    """Classify medication action (start/stop/switch) from the decision description."""
    prompt = load_prompt("phase_d_action.txt").format(decision=desc, drug="")
    raw = call_llm(client, [{"role": "user", "content": prompt}], **model_kwargs) \
              .choices[0].message.content.strip()
    action = _parse_json(raw).get("action", "")
    return action if action in VALID_ACTIONS else "start"


def _llm_classify(client, desc: str, model_kwargs: dict) -> dict:
    """
    Full type + action classification for a plan GT with no anchor events.
    Returns {"type": ..., "action": ...} — action present only when type is medication.
    """
    prompt = load_prompt("phase_d.txt").format(decision=desc)
    raw = call_llm(client, [{"role": "user", "content": prompt}], **model_kwargs) \
              .choices[0].message.content.strip()
    result = _parse_json(raw)
    t = result.get("type", "plan")
    if t not in VALID_TYPES:
        t = "plan"
    out: dict = {"type": t}
    if t == "medication":
        action = result.get("action", "")
        out["action"] = action if action in VALID_ACTIONS else _llm_action(client, desc, model_kwargs)
    return out


def _is_covered(client, plan_desc: str, anchored_gts: list, model_kwargs: dict) -> bool:
    """Check if a plan description is already covered by anchored GTs."""
    if not anchored_gts:
        return False
    
    anchored_text = "\n".join([f"- [{g['type']}] {g['description']}" for g in anchored_gts])
    prompt = f"""You are a clinical logic auditor.
A physician made a general management plan: "{plan_desc}"
In the same stage, we have already identified the following specific anchored actions:
{anchored_text}

Is the general plan "{plan_desc}" completely redundant or already covered by the specific actions listed above?
Example: "Started antibiotic therapy" is covered by "Started Vancomycin".
Example: "Managed pain" is covered by "Started Dilaudid PCA".
Example: "Low sodium diet" is NOT covered by "Started Lisinopril".

Output JSON only:
{{"covered": true|false}}
"""
    try:
        raw = call_llm(client, [{"role": "user", "content": prompt}], **model_kwargs) \
                  .choices[0].message.content.strip()
        # Simple JSON extraction
        import json
        import re
        m = re.search(r"\{.*\}", raw.replace("\n", ""))
        return json.loads(m.group()).get("covered", False)
    except:
        return False


# ------------------------------------------------------------------ #
# Phase D main                                                         #
# ------------------------------------------------------------------ #

def phase_d(client, stages: list, timeline: list,
            verbose: bool = True, model: str | None = None) -> list:
    """
    Convert each stage's single `gt` dict into a `gts` list of typed GT items.

    Each GT item: {type, description, event_index?, action? (medication only), primary?}
    """
    if verbose:
        print("\n========== PHASE D ==========")

    evt_by_idx = {e["index"]: e for e in timeline}
    model_kwargs = {"model": model} if model else {}

    for s in stages:
        gt = s.pop("gt")
        
        # Special case: diagnosis_list from scanner
        if gt.get("type") == "diagnosis_list":
            s["gts"] = [
                {
                    "type":        "diagnosis",
                    "description": f"[{d['icd_code']}] {d.get('long_title_diagnosis', d.get('long_title', ''))}",
                    "icd_code":    d["icd_code"],
                    "icd_version": d.get("icd_version", 9),
                    "primary":     True,
                }
                for d in gt.get("diagnoses", [])
            ]
            if verbose:
                print(f'  [diagnosis_list] -> {len(s["gts"])} items')
            continue

        desc = gt.get("description", "")

        # Collect anchor event indices
        if gt.get("event_index") is not None:
            anchor_indices = [gt["event_index"]]
        elif gt.get("event_range"):
            anchor_indices = list(range(gt["event_range"][0], gt["event_range"][1] + 1))
        else:
            anchor_indices = []

        # No anchor → fallback classify
        if not anchor_indices:
            item = _llm_classify(client, desc, model_kwargs)
            item["description"] = desc
            s["gts"] = [item]
            if verbose:
                a = f'({item["action"]})' if "action" in item else ""
                print(f'  [plan→{item["type"]}{a}] {desc[:80]}')
            continue

        # Build events text for the prompt
        events_text = "\n".join(
            f"  [{idx}] {fmt_event_label(evt_by_idx[idx])}"
            for idx in anchor_indices
            if idx in evt_by_idx
        )

        prompt = load_prompt("phase_d_classify.txt").format(decision=desc, events=events_text)
        raw = call_llm(client, [{"role": "user", "content": prompt}], **model_kwargs) \
                  .choices[0].message.content.strip()
        raw_gts = _parse_json(raw).get("gts", [])

        gts = []
        for g in raw_gts:
            if not isinstance(g, dict):
                continue
            t = g.get("type", "plan")
            if t not in VALID_TYPES:
                t = "plan"
            item: dict = {"type": t, "description": desc}
            if g.get("event_index") is not None:
                item["event_index"] = g["event_index"]
            if g.get("primary") is not None:
                item["primary"] = g["primary"]
            if t == "medication":
                action = g.get("action", "")
                item["action"] = action if action in VALID_ACTIONS else _llm_action(client, desc, model_kwargs)
            gts.append(item)

        # Fallback: LLM returned nothing
        if not gts:
            item = _llm_classify(client, desc, model_kwargs)
            item["description"] = desc
            gts = [item]

        # Dedup pass 1: drop un-anchored GTs whose (type, action) already has an anchored version
        has_anchored = {(g.get("type"), g.get("action")) for g in gts if g.get("event_index") is not None}
        gts = [g for g in gts
               if g.get("event_index") is not None
               or (g.get("type"), g.get("action")) not in has_anchored]

        # Dedup pass 2: intelligent coverage check for un-anchored plan GTs
        anchored_gts = [g for g in gts if g.get("event_index") is not None]
        unanchored_plans = [g for g in gts if g.get("event_index") is None and g.get("type") == "plan"]
        
        if unanchored_plans and anchored_gts:
            final_gts = [g for g in gts if g not in unanchored_plans]
            for up in unanchored_plans:
                if not _is_covered(client, up["description"], anchored_gts, model_kwargs):
                    final_gts.append(up)
            gts = final_gts

        s["gts"] = gts

        if verbose:
            for g in gts:
                p = "" if len(gts) == 1 else (" (primary)" if g.get("primary") else " (secondary)")
                a = f'/{g["action"]}' if "action" in g else ""
                print(f'  [{g.get("type", "?")}]{a}{p} {desc[:70]}')

    # Enrich GTs with structured fields from raw EHR events
    stages = enrich_gt(stages, timeline, verbose=verbose)
    return stages


# ------------------------------------------------------------------ #
# GT Enrichment (pure code, no LLM)                                    #
# ------------------------------------------------------------------ #

def enrich_gt(stages: list, timeline: list, verbose: bool = True) -> list:
    """
    Enrich each GT item with structured fields extracted directly from
    the raw EHR timeline events. Zero LLM calls, zero hallucination.

    For medication GTs:
        - drug_name, dose, dose_unit, route, frequency, prod_strength
    For procedure GTs:
        - icd_code, icd_version, procedure_title
    For diagnosis GTs:
        - (already has ICD code in description, no change needed)
    """
    if verbose:
        print("\n---------- ENRICH GT ----------")

    evt_by_idx = {e["index"]: e for e in timeline}

    for s in stages:
        for g in s.get("gts", []):
            gt_type = g.get("type")
            idx = g.get("event_index")

            # Collect all relevant event indices
            indices = []
            if idx is not None:
                indices.append(idx)
            if g.get("event_range"):
                r = g["event_range"]
                indices.extend(range(r[0], r[1] + 1))

            if not indices:
                continue

            # Pick the most representative event for field extraction
            ev = evt_by_idx.get(indices[0])
            if not ev:
                continue

            src = ev.get("source_table", "")

            if gt_type == "medication":
                # Try prescriptions first, then pharmacy, then emar_detail
                if src == "hosp_prescriptions_df":
                    g["drug_name"] = ev.get("drug", "")
                    g["dose"] = ev.get("dose_val_rx", "")
                    g["dose_unit"] = ev.get("dose_unit_rx", "")
                    g["route"] = ev.get("route", "")
                    g["prod_strength"] = ev.get("prod_strength", "")
                    g["frequency"] = ev.get("frequency", "")
                elif src == "hosp_pharmacy_df":
                    g["drug_name"] = ev.get("medication", "")
                    g["route"] = ev.get("route", "")
                    g["frequency"] = ev.get("frequency", "")
                    g["status"] = ev.get("status", "")
                elif src == "hosp_emar_detail_df":
                    g["drug_name"] = ev.get("medication", "")
                    g["route"] = ev.get("route", "")
                else:
                    # Fallback: scan other indices for a prescription/pharmacy/emar event
                    for alt_idx in indices[1:]:
                        alt_ev = evt_by_idx.get(alt_idx)
                        if not alt_ev:
                            continue
                        alt_src = alt_ev.get("source_table", "")
                        if alt_src == "hosp_prescriptions_df":
                            g["drug_name"] = alt_ev.get("drug", "")
                            g["dose"] = alt_ev.get("dose_val_rx", "")
                            g["dose_unit"] = alt_ev.get("dose_unit_rx", "")
                            g["route"] = alt_ev.get("route", "")
                            break
                        elif alt_src == "hosp_pharmacy_df":
                            g["drug_name"] = alt_ev.get("medication", "")
                            g["route"] = alt_ev.get("route", "")
                            break
                        elif alt_src == "hosp_emar_detail_df":
                            g["drug_name"] = alt_ev.get("medication", "")
                            g["route"] = alt_ev.get("route", "")
                            break

                if verbose and g.get("drug_name"):
                    print(f"  [{g.get('action','?')}] {g['drug_name']} "
                          f"{g.get('dose','')} {g.get('dose_unit','')} "
                          f"route={g.get('route','')}")

            elif gt_type in ["procedure", "diagnosis"]:
                # Robust ICD extraction
                icd = ev.get("icd_code")
                version = ev.get("icd_version", 9)
                title = ev.get("long_title_procedure") or ev.get("long_title_diagnosis") or ev.get("long_title") or ""

                # If missing from fields, try parsing label
                if not icd:
                    label = ev.get("label", "")
                    match = re.search(r"\[([A-Z0-9\.]+)\]", label)
                    if match:
                        icd = match.group(1).strip()
                
                if icd:
                    g["icd_code"] = icd
                    g["icd_version"] = version
                    if not g.get("procedure_title") and not g.get("diagnosis_title"):
                        key = "procedure_title" if gt_type == "procedure" else "diagnosis_title"
                        g[key] = title

                # Fallback: scan other indices
                if not g.get("icd_code"):
                    for alt_idx in indices[1:]:
                        alt_ev = evt_by_idx.get(alt_idx)
                        if not alt_ev: continue
                        icd = alt_ev.get("icd_code")
                        if not icd:
                            m = re.search(r"\[([A-Z0-9\.]+)\]", alt_ev.get("label", ""))
                            if m: icd = m.group(1).strip()
                        if icd:
                            g["icd_code"] = icd
                            g["icd_version"] = alt_ev.get("icd_version", 9)
                            break

                if verbose and g.get("icd_code"):
                    print(f"  [{gt_type}] ICD{g.get('icd_version','')}: "
                          f"{g['icd_code']} - {title[:60]}")

    return stages
