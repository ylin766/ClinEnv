"""env — clinical decision-making environment.

Public interface (consumed by evaluation/ and external runners):

    from env import run_episode

    episode_log = run_episode(case, model="gpt-5.4-mini-2026-03-17", verbose=True)

Episode log schema
------------------
{
    "subject_id": str,
    "hadm_id":    str,
    "model":      str,
    "stages": [
        {
            "label":         str,
            "context_range": [int, int],
            "turns":         int,
            "submissions":   [{"type": str, ...}],
            "gts":           [{"type": str, ...}],
            "messages":      [...]
        },
        ...
    ]
}
"""

from env.runtime.environment_controller import run_episode
from schema import StageResult, Submission, GTItem, Mode

__all__ = ["run_episode", "StageResult", "Submission", "GTItem", "Mode"]
