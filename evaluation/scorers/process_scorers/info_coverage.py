"""Information coverage scorer.

Per-speaker: LLM extracts key items from readview, marks which were received.
  coverage  = covered / total key items
  efficiency = coverage / n_info_calls
"""

from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from env.config_loader import get_config
from env.llm_client import chat_complete
from evaluation.scorers._openai_utils import get_client, get_model
from evaluation.scorers._usage import usage as _usage, progress as _progress
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

    def _score_one_speaker(speaker: str) -> tuple[str, dict] | None:
        rv = readviews.get(speaker, {})
        if not rv:
            return None

        available_info = json.dumps(rv, ensure_ascii=False)
        responses      = speaker_responses(messages, speaker)
        queried_text   = "\n---\n".join(responses) if responses else "(none)"

        user_msg = (
            f"AVAILABLE_INFO:\n{available_info}\n\n"
            f"QUERIED_RESPONSES:\n{queried_text}"
        )

        stage_label = stage_case.get("label", "unknown_stage")
        hadm_id     = stage_case.get("hadm_id", "unknown_hadm")
        print(f"  [info_coverage] Sending prompt to GPT for case {hadm_id}, stage {stage_label}, speaker {speaker}... (Input length: {len(user_msg)} chars)", flush=True)

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
            tok = resp.usage.total_tokens
            _usage.track("info_coverage",
                         prompt_tokens=resp.usage.prompt_tokens,
                         completion_tokens=resp.usage.completion_tokens)
        else:
            tok = "?"
        try:
            data = json.loads(resp.choices[0].message.content)
        except (json.JSONDecodeError, AttributeError):
            data = {}
        key_items = data.get("key_items", [])
        total     = len(key_items)
        covered   = sum(1 for item in key_items if item.get("covered"))
        coverage  = round(covered / total, 3) if total else 0.0
        print(_progress.tick(f"{hadm_id}  {stage_label}  {speaker}  cov={coverage:.2f}  tok={tok}"), flush=True)
        return speaker, {
            "total":    total,
            "covered":  covered,
            "coverage": coverage,
            "items":    key_items,
        }

    per_speaker: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=len(speakers)) as spk_pool:
        futures = {spk_pool.submit(_score_one_speaker, spk): spk for spk in speakers}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                spk, data = result
                per_speaker[spk] = data

    coverages        = [v["coverage"] for v in per_speaker.values() if "coverage" in v]
    coverage_overall = round(sum(coverages) / len(coverages), 3) if coverages else 0.0

    # Total items across all speakers (reflects stage info complexity)
    total_items = sum(v.get("total", 0) for v in per_speaker.values())
    # efficiency = coverage / max(1, K/N)
    # Scale-invariant: penalises over-querying (K>N) proportionally; no penalty when K≤N.
    if total_items > 0 and n_info_calls > 0:
        efficiency = round(coverage_overall / max(1.0, n_info_calls / total_items), 3)
    else:
        efficiency = coverage_overall

    return {
        "per_speaker":      per_speaker,
        "coverage_overall": coverage_overall,
        "n_info_calls":     n_info_calls,
        "efficiency":       efficiency,
    }
