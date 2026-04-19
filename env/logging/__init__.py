"""env.logging — episode log persistence.

Scoring lives in evaluation/ (a separate, independent module).
"""

from __future__ import annotations
import json
from pathlib import Path


def save_episode_log(log_path: str | Path, payload: dict) -> Path:
    """Write an episode log dict to disk as JSON.

    Args:
        log_path: Destination file path (.json).
        payload:  Episode log dict produced by env.runtime.run_episode,
                  or a scored log produced by evaluation.scorer.score_episode.

    Returns:
        Path to the written file.
    """
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return p
