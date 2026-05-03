"""Export all evaluation data to a single Excel file.

每行 = 一个 (subject_id, hadm_id, model, stage_index) 的完整记录。
排序：case → 固定模型顺序 → stage顺序。
所有图表均可从此表派生。

用法:
    python3 scripts/export_results_table.py
输出:
    results/benchmark_results.xlsx
"""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).parent.parent
CASES_DIR  = ROOT / "data" / "cases"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 固定模型顺序（能力从强到弱）
MODEL_ORDER = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]


def load_rows() -> list[dict]:
    rows = []

    for sid_dir in sorted(CASES_DIR.iterdir()):
        if not sid_dir.is_dir():
            continue
        for hid_dir in sorted(sid_dir.iterdir()):
            case_path = hid_dir / "case" / "case.json"
            if not case_path.exists():
                continue

            case_data = json.loads(case_path.read_text())

            for model in MODEL_ORDER:
                slug     = model.replace("/", "-").replace(":", "-")
                out_path = hid_dir / "runs" / slug / "output.json"
                ev_path  = hid_dir / "evals" / model / "latest.json"

                if not out_path.exists():
                    continue

                output  = json.loads(out_path.read_text())
                eval_d  = json.loads(ev_path.read_text()) if ev_path.exists() else {}

                out_stages  = output.get("stages", [])
                ev_stages   = eval_d.get("stages", [])
                case_stages = case_data.get("stages", [])

                total_stages_in_case = len(out_stages)

                for stage_index, out_s in enumerate(out_stages):
                    ev_s    = ev_stages[stage_index]   if stage_index < len(ev_stages)   else {}
                    case_s  = case_stages[stage_index] if stage_index < len(case_stages) else {}

                    gts  = out_s.get("gts", [])
                    subs = out_s.get("submissions", [])
                    sc   = ev_s.get("outcome_score", {})
                    ps   = ev_s.get("process_score", {})
                    ic   = ps.get("info_coverage", {})
                    lc   = ps.get("lab_cost", {})
                    mc   = ps.get("medication_cost", {})

                    # ── Stage context range ───────────────────────────
                    ctx       = out_s.get("context_range", [None, None])
                    ctx_start = ctx[0] if ctx else None
                    ctx_end   = ctx[1] if ctx else None
                    ctx_len   = (ctx_end - ctx_start) if (ctx_start is not None and ctx_end is not None) else None

                    # ── Ground Truth breakdown ────────────────────────
                    n_gt_total      = len(gts)
                    n_gt_primary    = sum(1 for g in gts if g.get("primary", True))
                    n_gt_diagnosis  = sum(1 for g in gts if g.get("type") == "diagnosis")
                    n_gt_medication = sum(1 for g in gts if g.get("type") == "medication")
                    n_gt_procedure  = sum(1 for g in gts if g.get("type") == "procedure")

                    type_counts   = {"diagnosis": n_gt_diagnosis, "medication": n_gt_medication, "procedure": n_gt_procedure}
                    dominant_gt_type = max(type_counts, key=type_counts.get) if n_gt_total > 0 else ""

                    gt_med_actions = [g.get("action", "") for g in gts if g.get("type") == "medication"]
                    n_gt_medication_action_start  = gt_med_actions.count("start")
                    n_gt_medication_action_stop   = gt_med_actions.count("stop")
                    n_gt_medication_action_adjust = gt_med_actions.count("adjust")
                    n_gt_medication_action_switch = gt_med_actions.count("switch")

                    # ── Submission breakdown ──────────────────────────
                    sub_med_actions = [s.get("action", "") for s in subs if s.get("type") == "medication"]
                    n_submitted_medication_action_start  = sub_med_actions.count("start")
                    n_submitted_medication_action_stop   = sub_med_actions.count("stop")
                    n_submitted_medication_action_adjust = sub_med_actions.count("adjust")
                    n_submitted_medication_action_switch = sub_med_actions.count("switch")

                    # ── Match-level analysis ──────────────────────────
                    matches = sc.get("matches", [])

                    diag_match_scores = [m["score"] for m in matches if m["gt_item"].get("type") == "diagnosis"]
                    med_match_scores  = [m["score"] for m in matches if m["gt_item"].get("type") == "medication"]
                    proc_match_scores = [m["score"] for m in matches if m["gt_item"].get("type") == "procedure"]

                    avg_diagnosis_match_score  = sum(diag_match_scores) / len(diag_match_scores) if diag_match_scores else None
                    avg_medication_match_score = sum(med_match_scores)  / len(med_match_scores)  if med_match_scores  else None
                    avg_procedure_match_score  = sum(proc_match_scores) / len(proc_match_scores) if proc_match_scores else None

                    n_medication_action_correct = sum(
                        1 for m in matches
                        if m["gt_item"].get("type") == "medication"
                        and m["gt_item"].get("action", "") == m["submission"].get("action", "")
                        and m["gt_item"].get("action", "") != ""
                    )
                    n_medication_action_wrong = sum(
                        1 for m in matches
                        if m["gt_item"].get("type") == "medication"
                        and m["gt_item"].get("action", "") != m["submission"].get("action", "")
                        and m["gt_item"].get("action", "") != ""
                        and m["submission"].get("action", "") != ""
                    )

                    # ── Outcome scores ────────────────────────────────
                    outcome_f1        = sc.get("f1")
                    outcome_recall    = sc.get("recall")
                    outcome_precision = sc.get("precision")
                    outcome_hits      = sc.get("hits")
                    outcome_total_gt  = sc.get("total")

                    # ── Process: Information Coverage ─────────────────
                    per_speaker = ic.get("per_speaker", {})

                    def _safe_get(spk: str, field: str):
                        v = per_speaker.get(spk, {})
                        if "error" in v:
                            return None
                        return v.get(field)

                    info_coverage_nurse_fraction   = _safe_get("nurse",   "coverage")
                    info_coverage_patient_fraction = _safe_get("patient", "coverage")
                    info_coverage_lab_fraction     = _safe_get("lab",     "coverage")
                    info_total_items_nurse         = _safe_get("nurse",   "total")
                    info_total_items_patient       = _safe_get("patient", "total")
                    info_total_items_lab           = _safe_get("lab",     "total")
                    info_covered_items_nurse       = _safe_get("nurse",   "covered")
                    info_covered_items_patient     = _safe_get("patient", "covered")
                    info_covered_items_lab         = _safe_get("lab",     "covered")

                    info_coverage_overall_fraction = ic.get("coverage_overall")
                    info_number_of_query_calls     = ic.get("n_info_calls")
                    info_coverage_efficiency       = ic.get("efficiency")

                    # ── Process: Lab Cost ─────────────────────────────
                    lab_number_of_successful_queries = lc.get("n_found")
                    lab_number_of_failed_queries     = lc.get("n_not_found")
                    lab_total_number_of_queries      = lc.get("n_queries")
                    lab_wasted_query_ratio           = lc.get("wasted_ratio")
                    lab_wasted_cost_usd              = lc.get("wasted_cost_usd")
                    lab_total_estimated_cost_usd     = lc.get("total_cost_usd")
                    lab_query_efficiency_score       = ps.get("lab_efficiency")

                    # ── Process: Medication Cost ──────────────────────
                    medication_number_of_submissions    = mc.get("n_medications")
                    medication_total_estimated_daily_cost_usd = mc.get("total_daily_cost")

                    row = {
                        # ── Case & Stage Identity ─────────────────────
                        "subject_id":                              sid_dir.name,
                        "hospital_admission_id":                   hid_dir.name,
                        "model_name":                              model,
                        "stage_index":                             stage_index,
                        "stage_label":                             out_s.get("label", ""),
                        "total_stages_in_case":                    total_stages_in_case,

                        # ── Stage Context ─────────────────────────────
                        "stage_context_event_start":               ctx_start,
                        "stage_context_event_end":                 ctx_end,
                        "stage_context_event_span_length":         ctx_len,

                        # ── Ground Truth ──────────────────────────────
                        "ground_truth_total_items":                n_gt_total,
                        "ground_truth_primary_items":              n_gt_primary,
                        "ground_truth_diagnosis_items":            n_gt_diagnosis,
                        "ground_truth_medication_items":           n_gt_medication,
                        "ground_truth_procedure_items":            n_gt_procedure,
                        "ground_truth_dominant_type":              dominant_gt_type,
                        "ground_truth_medication_action_start":    n_gt_medication_action_start,
                        "ground_truth_medication_action_stop":     n_gt_medication_action_stop,
                        "ground_truth_medication_action_adjust":   n_gt_medication_action_adjust,
                        "ground_truth_medication_action_switch":   n_gt_medication_action_switch,

                        # ── Submissions ───────────────────────────────
                        "submitted_medication_action_start":       n_submitted_medication_action_start,
                        "submitted_medication_action_stop":        n_submitted_medication_action_stop,
                        "submitted_medication_action_adjust":      n_submitted_medication_action_adjust,
                        "submitted_medication_action_switch":      n_submitted_medication_action_switch,
                        "medication_action_correct_count":         n_medication_action_correct,
                        "medication_action_wrong_count":           n_medication_action_wrong,

                        # ── Outcome Scores ────────────────────────────
                        "outcome_f1_score":                        outcome_f1,
                        "outcome_recall_score":                    outcome_recall,
                        "outcome_precision_score":                 outcome_precision,
                        "outcome_weighted_hits":                   outcome_hits,
                        "outcome_total_ground_truth_weight":       outcome_total_gt,
                        "outcome_avg_diagnosis_match_score":       avg_diagnosis_match_score,
                        "outcome_avg_medication_match_score":      avg_medication_match_score,
                        "outcome_avg_procedure_match_score":       avg_procedure_match_score,

                        # ── Process: Information Coverage ─────────────
                        "info_coverage_overall_fraction":          info_coverage_overall_fraction,
                        "info_number_of_query_calls":              info_number_of_query_calls,
                        "info_coverage_efficiency_score":          info_coverage_efficiency,
                        "info_coverage_nurse_fraction":            info_coverage_nurse_fraction,
                        "info_coverage_patient_fraction":          info_coverage_patient_fraction,
                        "info_coverage_lab_fraction":              info_coverage_lab_fraction,
                        "info_total_available_items_nurse":        info_total_items_nurse,
                        "info_total_available_items_patient":      info_total_items_patient,
                        "info_total_available_items_lab":          info_total_items_lab,
                        "info_covered_items_nurse":                info_covered_items_nurse,
                        "info_covered_items_patient":              info_covered_items_patient,
                        "info_covered_items_lab":                  info_covered_items_lab,

                        # ── Process: Lab Query Cost ───────────────────
                        "lab_number_of_successful_queries":        lab_number_of_successful_queries,
                        "lab_number_of_failed_queries":            lab_number_of_failed_queries,
                        "lab_total_number_of_queries":             lab_total_number_of_queries,
                        "lab_wasted_query_ratio":                  lab_wasted_query_ratio,
                        "lab_wasted_cost_usd":                     lab_wasted_cost_usd,
                        "lab_total_estimated_cost_usd":            lab_total_estimated_cost_usd,
                        "lab_query_efficiency_score":              lab_query_efficiency_score,

                        # ── Process: Medication Cost ──────────────────
                        "medication_number_of_submissions":        medication_number_of_submissions,
                        "medication_total_estimated_daily_cost_usd": medication_total_estimated_daily_cost_usd,
                    }
                    rows.append(row)

    return rows


def main():
    print("读取数据...")
    rows = load_rows()

    df = pd.DataFrame(rows)

    # 按 case → 固定模型顺序 → stage 排序
    model_cat = pd.Categorical(df["model_name"], categories=MODEL_ORDER, ordered=True)
    df = df.assign(model_name=model_cat)
    df = df.sort_values(
        ["subject_id", "hospital_admission_id", "model_name", "stage_index"]
    ).reset_index(drop=True)

    print(f"共 {len(df)} 行 × {len(df.columns)} 列")
    print(f"模型分布:\n{df['model_name'].value_counts().to_string()}")
    print(f"有process数据的行: {df['info_coverage_overall_fraction'].notna().sum()}")

    out_path = OUTPUT_DIR / "benchmark_results.xlsx"
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"\n已写入: {out_path}")


if __name__ == "__main__":
    main()
