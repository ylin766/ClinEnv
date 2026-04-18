"""History recall tool.
Only responsible for returning past admission information in two layers: Summary layer / Detail layer.
"""


def recall_history(history_view, hadm_id: str | None = None) -> str:
    """Returns summary if hadm_id is empty, returns details otherwise."""
    raise NotImplementedError
