"""env — clinical decision-making environment.

Public interface (consumed by evaluation/ and external runners):

    from env import run_episode, save_episode_log

    episode_log = run_episode(case, model="gpt-4o", verbose=True)
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
from schema import StageResult, Submission, GTItem

__all__ = ["run_episode", "save_episode_log", "StageResult", "Submission", "GTItem"]
