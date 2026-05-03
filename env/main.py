"""Environment execution entry point.

Runs one episode and writes:
  data/cases/<subject_id>/<hadm_id>/runs/<model>/output.json   — submissions + GTs (no scores)
  data/cases/<subject_id>/<hadm_id>/runs/<model>/dialogue.json — full message history (interactive only)

Scoring is handled separately by evaluation/main.py.

Usage:
    python -m env.main --subject-id 10196757 --hadm-id 24725711
    python -m env.main --manifest data/cases/manifest.jsonl --index 0
    python -m env.main --manifest data/cases/manifest.jsonl --index 0 --model gpt-5.4-mini
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

from env import run_episode
from env.config_loader import get_config
from env.readers.prepared_case_reader import load_prepared_case, load_from_manifest

_DATA_DIR  = Path(__file__).parent.parent / "data"
_CASES_DIR = _DATA_DIR / "cases"


def main() -> int:
    conf = get_config("runtime")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject-id", help="Subject ID (use with --hadm-id)")
    group.add_argument("--manifest",   help="Manifest JSONL to pick a case from")
    parser.add_argument("--hadm-id",   help="Admission ID (use with --subject-id)")
    parser.add_argument("--index",     type=int, default=0, help="Case index in manifest (default: 0)")
    parser.add_argument("--model",     default=conf.get("mut_model"), help="Model for MUT")
    parser.add_argument("--env-model", default=conf.get("env_model", "gpt-5.4-mini"), help="Model for env agents")
    parser.add_argument("--mode",      default=conf.get("mode", "interactive"),
                        choices=["direct", "interactive"])
    parser.add_argument("--quiet",     action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    import os
    if args.env_model:
        os.environ["ENV_MODEL"] = args.env_model

    if args.subject_id:
        if not args.hadm_id:
            print("--hadm-id required with --subject-id", file=sys.stderr)
            return 1
        case = load_prepared_case(args.subject_id, args.hadm_id)
    else:
        case = load_from_manifest(Path(args.manifest), args.index)

    if verbose:
        print(f"Loaded case: subject={case['subject_id']} hadm={case['hadm_id']} stages={len(case['stages'])}")

    episode_log = run_episode(
        case,
        model=args.model,
        mode=args.mode,
        verbose=verbose,
        hint_counts=conf.get("runtime_settings", {}).get("hint_counts", True),
    )

    subject_id = str(case["subject_id"])
    hadm_id    = str(case["hadm_id"])
    model      = episode_log.get("model", args.model or "default")
    mode       = args.mode
    model_slug = model.replace("/", "-").replace(":", "-")
    run_at     = datetime.now(timezone.utc).isoformat()

    output_dir = _CASES_DIR / subject_id / hadm_id / "runs" / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── output.json: env run result only, no scores ───────────────────
    output = {
        "subject_id": subject_id,
        "hadm_id":    hadm_id,
        "model":      model,
        "mode":       mode,
        "run_at":     run_at,
        "stages": [
            {
                "label":         s.get("label", f"stage_{i}"),
                "context_range": s["context_range"],
                "gts":           s.get("gts", []),
                "submissions":   s.get("submissions", []),
            }
            for i, s in enumerate(episode_log["stages"])
        ],
    }
    output_path = output_dir / "output.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── dialogue.json: full message history (interactive only) ────────
    if mode == "interactive":
        dialogue = {
            "subject_id": subject_id,
            "hadm_id":    hadm_id,
            "model":      model,
            "mode":       mode,
            "run_at":     run_at,
            "stages": [
                {
                    "label":         s.get("label", f"stage_{i}"),
                    "context_range": s["context_range"],
                    "messages":      s.get("messages", []),
                }
                for i, s in enumerate(episode_log["stages"])
            ],
        }
        dialogue_path = output_dir / "dialogue.json"
        dialogue_path.write_text(json.dumps(dialogue, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Episode complete.  run_at={run_at}")
        print(f"Output:   {output_path}")
        if mode == "interactive":
            print(f"Dialogue: {output_dir / 'dialogue.json'}")
        print(f"\nRun evaluation:  python -m evaluation.main "
              f"--subject-id {subject_id} --hadm-id {hadm_id} --model {model_slug}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
