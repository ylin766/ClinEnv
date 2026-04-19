"""note_section GT scoring — LLM binary judge.

note_section GT items represent clinical decisions that cannot be expressed
as an ICD code, procedure, or drug name (e.g. "decided to stop Lovenox").

The judge receives the GT decision span and all submissions from the stage,
then determines whether any submission captures the core decision (0 or 1).
"""

from __future__ import annotations
import json
from pathlib import Path

from schema import Submission
from evaluation.scorers._openai_utils import get_client, get_model

_PROMPT_FILE = Path(__file__).parent.parent.parent / "prompts" / "scoring" / "note_judge.txt"


def score_note_section(gt_span: str, all_submissions: list[Submission]) -> dict:
    """Judge whether any submission captures the GT note_section decision.

    Args:
        gt_span:         The ground-truth clinical decision text.
        all_submissions: All submissions from the stage (any type).

    Returns:
        {"score": 0.0|1.0, "matched_submission": str|None, "reason": str}
    """
    if not all_submissions:
        return {"score": 0.0, "matched_submission": None, "reason": "no submissions"}

    system_prompt = _PROMPT_FILE.read_text(encoding="utf-8").strip()

    sub_lines = []
    for i, s in enumerate(all_submissions, 1):
        sub_lines.append(f"{i}. [{s.get('type', '?')}] {s.get('value', '?')}")

    user_msg = (
        f"Ground-truth clinical decision:\n\"{gt_span}\"\n\n"
        f"Physician's submissions:\n" + "\n".join(sub_lines)
    )

    try:
        resp = get_client().chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_completion_tokens=80,
        )
        data = json.loads(resp.choices[0].message.content.strip())
        score = float(data.get("score", 0))
        reason = data.get("reason", "")
        idx = data.get("matched")
        matched = None
        if idx is not None:
            try:
                matched = all_submissions[int(idx) - 1].get("value")
            except (IndexError, ValueError):
                pass
        return {"score": score, "matched_submission": matched, "reason": reason}
    except Exception as e:
        return {"score": 0.0, "matched_submission": None, "reason": f"judge error: {e}"}
