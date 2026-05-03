"""Prepared case schema definition and validation.

Validates planner output only — readviews are built by readview_builder.py
and are always structurally correct, so they are not checked here.
"""

_VALID_GT_TYPES         = {"procedure", "diagnosis", "medication"}
_VALID_MED_ACTIONS      = {"start", "stop", "adjust", "switch"}


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

        for key in ("context_range", "gts"):
            if key not in stage:
                errors.append(_err(p, f"missing key '{key}'"))

        # context_range
        ir = stage.get("context_range")
        if not (isinstance(ir, list) and len(ir) == 2 and
                isinstance(ir[0], int) and isinstance(ir[1], int) and ir[0] <= ir[1]):
            errors.append(_err(p + ".context_range", "must be [int, int] with start <= end"))
        else:
            if i > 0 and ir[0] < prev_end:
                errors.append(_err(p + ".context_range", f"overlap: expected start >= {prev_end}, got {ir[0]}"))
            prev_end = ir[1]

        # gts items
        gts = stage.get("gts", [])
        if not isinstance(gts, list):
            errors.append(_err(p + ".gts", "must be a list"))
        else:
            stage_end = ir[1] if isinstance(ir, list) and len(ir) == 2 else 0
            for j, item in enumerate(gts):
                gp = f"{p}.gts[{j}]"
                gt_type = item.get("type")
                if gt_type not in _VALID_GT_TYPES:
                    errors.append(_err(gp, f"type must be one of {_VALID_GT_TYPES}, got {gt_type!r}"))
                elif gt_type in ("procedure", "diagnosis"):
                    # allow icd_code to be optional if it's a generated summary, but usually required
                    if "icd_code" not in item and "description" not in item:
                        errors.append(_err(gp, "missing 'icd_code' or 'description'"))
                elif gt_type == "medication":
                    action = item.get("action")
                    if action and action != "null" and action not in _VALID_MED_ACTIONS:
                        errors.append(_err(gp, f"action must be one of {_VALID_MED_ACTIONS}, got {action!r}"))
                    if "drug_name" not in item and "description" not in item:
                        errors.append(_err(gp, "missing 'drug_name' or 'description'"))
                    
    return errors
