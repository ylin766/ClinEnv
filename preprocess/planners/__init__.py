from .runner import run

def plan_decision_points(record, verbose: bool = True, model: str | None = None) -> dict:
    """
    Adapter function that wraps the new planner runner to match the old planner interface.
    Returns a dict with 'stages' mapped to the new planner's output.
    """
    stages = run(
        subject_id=record.subject_id,
        hadm_id=record.hadm_id,
        verbose=verbose,
        model=model
    )
    return {
        "subject_id": record.subject_id,
        "hadm_id": record.hadm_id,
        "stages": stages
    }
