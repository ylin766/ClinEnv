"""env — clinical decision-making environment.

Public interface (consumed by evaluation/ and external runners):

    from env import run_episode, save_episode_log

    episode_log = run_episode(case, model="gpt-5.4-mini-2026-03-17", verbose=True)
    save_episode_log("data/episodes/subj/hadm.json", episode_log)

Episode log schema
------------------
{
    "subject_id": str,
    "hadm_id":    str,
    "model":      str,
    "stages": [
        {
            "label":       str,
            "index_range": [int, int],
            "turns":       int,
            "submissions": [{"type": str, ...}],
            "gt":          [{"source": str, ...}],
            "messages":    [...]
        },
        ...
    ]
}
"""

from env.runtime.environment_controller import run_episode
from env.logging import save_episode_log
from schema import StageResult, Submission, GTItem, Mode

__all__ = ["run_episode", "save_episode_log", "StageResult", "Submission", "GTItem", "Mode"]
