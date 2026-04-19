"""evaluation — independent scoring module.

Consumes episode logs produced by env/ and computes scores.
Can be used with any episode log that follows the env output schema.

Public interface:

    from evaluation import score_episode, score_stage

    scored = score_episode(episode_log)   # adds 'score' to each stage + overall_score
    score  = score_stage(stage_log)       # score a single stage in isolation
"""

from evaluation.scorer import score_episode, score_stage

__all__ = ["score_episode", "score_stage"]
