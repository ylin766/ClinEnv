"""Information coverage scorer.

Per-speaker: LLM extracts key items from readview, marks which were received.
  coverage  = covered / total key items
  efficiency = coverage / n_info_calls
"""

from __future__ import annotations
import json
from pathlib import Path

from env.config_loader import get_config
from env.llm_client import chat_complete
from evaluation.scorers._openai_utils import get_client, get_model
from evaluation.scorers._usage import usage as _usage
from evaluation.scorers.process_scorers._dialogue import info_tool_results, speaker_responses

_PROMPT = Path(__file__).parent.parent.parent.parent / "prompts" / "scoring" / "info_coverage_judge.txt"


def score_info_coverage(stage_dialogue: dict, stage_case: dict) -> dict:
    """Score information coverage and efficiency for one stage.

    Args:
        stage_dialogue: one stage entry from dialogue.json (has 'messages')
        stage_case:     one stage entry from case.json     (has 'readviews')

    Returns dict with:
        per_speaker:      {speaker: {total, covered, coverage, items}}
        coverage_overall: float
        n_info_calls:     int
        efficiency:       coverage_overall / n_info_calls  (0 if no calls)
    """
    eval_conf = get_config("eval")
    speakers  = eval_conf.get("process_scorers", {}).get("info_coverage", {}).get(
        "speakers", ["nurse", "patient", "lab"]
    )

    messages  = stage_dialogue.get("messages", [])
    readviews = stage_case.get("readviews", {})
    n_info_calls = len(info_tool_results(messages))

    system_prompt = _PROMPT.read_text(encoding="utf-8").strip()
    client = get_client()
    model  = get_model()

    per_speaker: dict[str, dict] = {}

    for speaker in speakers:
        rv = readviews.get(speaker, {})
        if not rv:
            continue

        available_info = json.dumps(rv, ensure_ascii=False)[:4000]
        responses      = speaker_responses(messages, speaker)
        queried_text   = "\n---\n".join(responses) if responses else "(none)"

        user_msg = (
            f"AVAILABLE_INFO:\n{available_info}\n\n"
            f"QUERIED_RESPONSES:\n{queried_text}"
        )

        try:
            resp = chat_complete(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_completion_tokens=4096,
                response_format={"type": "json_object"},
            )
            if hasattr(resp, "usage") and resp.usage:
                _usage.track("info_coverage",
                             prompt_tokens=resp.usage.prompt_tokens,
                             completion_tokens=resp.usage.completion_tokens)
            data      = json.loads(resp.choices[0].message.content)
            key_items = data.get("key_items", [])
            total     = len(key_items)
            covered   = sum(1 for item in key_items if item.get("covered"))
            coverage  = round(covered / total, 3) if total else 0.0
            per_speaker[speaker] = {
                "total":    total,
                "covered":  covered,
                "coverage": coverage,
                "items":    key_items,
            }
        except Exception as e:
            per_speaker[speaker] = {"error": str(e), "coverage": 0.0}

    coverages        = [v["coverage"] for v in per_speaker.values() if "coverage" in v]
    coverage_overall = round(sum(coverages) / len(coverages), 3) if coverages else 0.0

    # Total items across all speakers (reflects stage info complexity)
    total_items = sum(v.get("total", 0) for v in per_speaker.values())
    # efficiency = coverage × N / (N + K)
    # Approaches coverage when K≪N; penalises over-querying when K≫N.
    if total_items > 0 and n_info_calls > 0:
        efficiency = round(coverage_overall * total_items / (total_items + n_info_calls), 3)
    else:
        efficiency = 0.0

    return {
        "per_speaker":      per_speaker,
        "coverage_overall": coverage_overall,
        "n_info_calls":     n_info_calls,
        "efficiency":       efficiency,
    }
