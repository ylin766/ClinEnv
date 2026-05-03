"""Prepared case writer.
Serializes a validated prepared case and planner plan to disk; maintains a manifest index.

File layout per case:
    data/cases/<subject_id>/<hadm_id>/case.json   — full prepared case (plan + readviews)
    data/cases/<subject_id>/<hadm_id>/plan.json   — planner output only (no readviews)
"""

import json
from pathlib import Path

from preprocess.schemas.prepared_case_schema import validate_prepared_case

_DEFAULT_OUT_DIR = Path(__file__).parent.parent.parent / "data" / "cases"

# Fields that belong to the planner plan (exclude readview-added fields)
_PLAN_STAGE_FIELDS = {"context_range", "gts"}


def write_plan(planner_output: dict, out_dir: Path = _DEFAULT_OUT_DIR) -> Path:
    """
    Write planner output (stages without readviews) to <hadm_id>_plan.json.
    Called before build_readviews so the file always reflects pure planner decisions.
    """
    subject_id = planner_output["subject_id"]
    hadm_id    = planner_output["hadm_id"]

    case_dir = out_dir / subject_id / hadm_id / "plan"
    case_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "subject_id": subject_id,
        "hadm_id":    hadm_id,
        "stages": [
            {k: v for k, v in stage.items() if k in _PLAN_STAGE_FIELDS}
            for stage in planner_output.get("stages", [])
        ],
    }

    plan_path = case_dir / "plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return plan_path


def write_prepared_case(payload: dict, out_dir: Path = _DEFAULT_OUT_DIR) -> Path:
    """
    Validate and write the full prepared case (plan + readviews) to <hadm_id>.json.

    Raises:
        ValueError if schema validation fails.
    """
    errors = validate_prepared_case(payload)
    if errors:
        raise ValueError("Prepared case failed validation:\n" + "\n".join(errors))

    subject_id = payload["subject_id"]
    hadm_id    = payload["hadm_id"]

    case_dir = out_dir / subject_id / hadm_id / "case"
    case_dir.mkdir(parents=True, exist_ok=True)

    out_path = case_dir / "case.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


def update_manifest(payload: dict, out_dir: Path = _DEFAULT_OUT_DIR) -> None:
    """
    Append or update an entry in data/cases/manifest.jsonl.
    Each line: {subject_id, hadm_id, n_stages, path}
    """
    manifest_path = out_dir / "manifest.jsonl"
    entry = {
        "subject_id": payload["subject_id"],
        "hadm_id":    payload["hadm_id"],
        "n_stages":   len(payload["stages"]),
        "path":       str(out_dir / payload["subject_id"] / payload["hadm_id"] / "case.json"),
    }

    entries = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                e = json.loads(line)
                entries[e["hadm_id"]] = e

    entries[entry["hadm_id"]] = entry

    manifest_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries.values()) + "\n",
        encoding="utf-8",
    )
