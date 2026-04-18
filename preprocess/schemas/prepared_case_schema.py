"""Prepared case schema definition and validation.

A prepared case is the serialized output of the preprocessing pipeline.
It contains the planner's decision stages enriched with per-agent readviews,
ready to be loaded by the runtime environment.

Schema
------
{
  "subject_id": str,
  "hadm_id":    str,
  "stages": [
    {
      "label":             str,
      "index_range":       [int, int],
      "trigger":           {"agent": "patient" | "nurse"},
      "decision_required": str,
      "gt_type":           "diagnosis" | "management_plan" | "discharge_plan",
      "gt": [
        {"source": "icd_diagnosis",  "icd_code": str, "icd_version": int, "display": str}
        {"source": "icd_procedure",  "index": int, "icd_code": str, "icd_version": int, "display": str}
        {"source": "medication",     "index": int, "drug": str}
        {"source": "note_section",   "section": str, "span": str}
      ],
      "readviews": {
        "patient": {
          "chief_complaint":      str | None,
          "hpi":                  str | None,
          "past_medical_history": str | None,
          "age":                  int | None,
          "gender":               str | None
        },
        "nurse": {"events": [...]},
        "lab":   {"events": [...]}
      }
    }
  ]
}
"""

_VALID_AGENTS    = {"patient", "nurse"}
_VALID_GT_TYPES  = {"diagnosis", "management_plan", "discharge_plan"}
_VALID_GT_SRC    = {"icd_diagnosis", "icd_procedure", "medication", "note_section"}
_VALID_AGENTS_RV = {"patient", "nurse", "lab"}


def _err(path: str, msg: str) -> str:
    return f"[{path}] {msg}"


def validate_prepared_case(payload) -> list[str]:
    """
    Validate a prepared case dict against the schema.

    Returns a list of error strings. Empty list means valid.
    """
    errors = []

    if not isinstance(payload, dict):
        return [_err("root", "must be a dict")]

    for key in ("subject_id", "hadm_id", "stages"):
        if key not in payload:
            errors.append(_err("root", f"missing required key '{key}'"))

    if "stages" not in payload:
        return errors

    stages = payload["stages"]
    if not isinstance(stages, list) or len(stages) == 0:
        errors.append(_err("stages", "must be a non-empty list"))
        return errors

    prev_end = -1
    for i, stage in enumerate(stages):
        p = f"stages[{i}]"

        # required keys
        for key in ("label", "index_range", "trigger", "decision_required", "gt", "gt_type", "readviews"):
            if key not in stage:
                errors.append(_err(p, f"missing key '{key}'"))

        # index_range
        ir = stage.get("index_range")
        if not (isinstance(ir, list) and len(ir) == 2 and
                isinstance(ir[0], int) and isinstance(ir[1], int) and ir[0] <= ir[1]):
            errors.append(_err(p + ".index_range", "must be [int, int] with start <= end"))
        else:
            if i == 0 and ir[0] != 0:
                errors.append(_err(p + ".index_range", "first stage must start at 0"))
            if i > 0 and ir[0] != prev_end + 1:
                errors.append(_err(p + ".index_range", f"gap or overlap: expected start {prev_end + 1}, got {ir[0]}"))
            prev_end = ir[1]

        # trigger
        trigger = stage.get("trigger", {})
        if not isinstance(trigger, dict) or trigger.get("agent") not in _VALID_AGENTS:
            errors.append(_err(p + ".trigger", f"agent must be one of {_VALID_AGENTS}"))

        # gt_type
        if stage.get("gt_type") not in _VALID_GT_TYPES:
            errors.append(_err(p + ".gt_type", f"must be one of {_VALID_GT_TYPES}"))

        # gt items
        gt = stage.get("gt", [])
        if not isinstance(gt, list) or len(gt) == 0:
            errors.append(_err(p + ".gt", "must be a non-empty list"))
        else:
            for j, item in enumerate(gt):
                gp = f"{p}.gt[{j}]"
                src = item.get("source")
                if src not in _VALID_GT_SRC:
                    errors.append(_err(gp, f"source must be one of {_VALID_GT_SRC}"))
                elif src == "icd_diagnosis":
                    for k in ("icd_code", "icd_version", "display"):
                        if k not in item:
                            errors.append(_err(gp, f"missing '{k}'"))
                elif src == "icd_procedure":
                    for k in ("index", "icd_code", "icd_version", "display"):
                        if k not in item:
                            errors.append(_err(gp, f"missing '{k}'"))
                elif src == "medication":
                    for k in ("index", "drug"):
                        if k not in item:
                            errors.append(_err(gp, f"missing '{k}'"))
                elif src == "note_section":
                    for k in ("section", "span"):
                        if k not in item:
                            errors.append(_err(gp, f"missing '{k}'"))

        # readviews
        rv = stage.get("readviews", {})
        if not isinstance(rv, dict):
            errors.append(_err(p + ".readviews", "must be a dict"))
        else:
            for agent in _VALID_AGENTS_RV:
                if agent not in rv:
                    errors.append(_err(p + f".readviews", f"missing agent '{agent}'"))

    return errors
