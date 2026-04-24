"""Prepared case schema definition and validation.

Validates planner output only — readviews are built by readview_builder.py
and are always structurally correct, so they are not checked here.

Schema (planner-produced fields)
---------------------------------
{
  "subject_id": str,
  "hadm_id":    str,
  "stages": [
    {
      "label":            str,
      "index_range":      [int, int],
      "trigger":          {"agent": "patient" | "nurse", "context": str},
      "available_agents": [str, ...],
      "gt": [
        {"type": "diagnosis",  "icd_code": str, "icd_version": int, "display": str}
        {"type": "procedure",  "index": int, "icd_code": str, "icd_version": int, "display": str}
        {"type": "medication", "action": "start"|"stop"|"increase_dose"|"decrease_dose",
                               "index": int (required for action="start" only), "drug": str}
        {"type": "plan",       "section": str, "span": str}
      ]
    }
  ]
}
"""

_VALID_TRIGGER_AGENTS   = {"patient", "nurse"}
_VALID_GT_TYPES         = {"procedure", "diagnosis", "medication", "plan"}
_VALID_MED_ACTIONS      = {"start", "stop", "increase_dose", "decrease_dose"}


def _err(path: str, msg: str) -> str:
    return f"[{path}] {msg}"


def validate_prepared_case(payload) -> list[str]:
    """
    Validate planner-produced fields of a prepared case.
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

        for key in ("label", "index_range", "trigger", "gt"):
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
        if not isinstance(trigger, dict) or trigger.get("agent") not in _VALID_TRIGGER_AGENTS:
            errors.append(_err(p + ".trigger", f"agent must be one of {_VALID_TRIGGER_AGENTS}"))

        # gt items
        gt = stage.get("gt", [])
        if not isinstance(gt, list) or len(gt) == 0:
            errors.append(_err(p + ".gt", "must be a non-empty list"))
        else:
            stage_end = ir[1] if isinstance(ir, list) and len(ir) == 2 else 0
            for j, item in enumerate(gt):
                gp = f"{p}.gt[{j}]"
                gt_type = item.get("type")
                if gt_type not in _VALID_GT_TYPES:
                    errors.append(_err(gp, f"type must be one of {_VALID_GT_TYPES}, got {gt_type!r}"))
                elif gt_type in ("procedure", "diagnosis"):
                    for k in ("icd_code", "icd_version", "display"):
                        if k not in item:
                            errors.append(_err(gp, f"missing '{k}'"))
                    if gt_type == "procedure":
                        if "index" not in item:
                            errors.append(_err(gp, "missing 'index'"))
                        elif item["index"] <= stage_end:
                            errors.append(_err(gp,
                                f"GT index {item['index']} must be > stage end {stage_end} "
                                f"(GT must not be visible within this stage's event window)"))
                elif gt_type == "medication":
                    action = item.get("action", "start")
                    if action not in _VALID_MED_ACTIONS:
                        errors.append(_err(gp, f"action must be one of {_VALID_MED_ACTIONS}, got {action!r}"))
                    if "drug" not in item:
                        errors.append(_err(gp, "missing 'drug'"))
                    # 'start' requires a timeline index; stop/increase/decrease do not
                    if action == "start":
                        if "index" not in item:
                            errors.append(_err(gp, "missing 'index' (required for action='start')"))
                        elif item["index"] <= stage_end:
                            errors.append(_err(gp,
                                f"GT index {item['index']} must be > stage end {stage_end} "
                                f"(GT must not be visible within this stage's event window)"))
                elif gt_type == "plan":
                    for k in ("section", "span"):
                        if k not in item:
                            errors.append(_err(gp, f"missing '{k}'"))

    return errors
