"""Environment execution entry point.

Usage:
    python -m env.main --subject-id 10196757 --hadm-id 24725711
    python -m env.main --manifest data/cases/manifest.jsonl --index 0
    python -m env.main --manifest data/cases/manifest.jsonl --index 0 --model gpt-5.4-mini-2026-03-17
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

import json

from env import run_episode
from env.readers.prepared_case_reader import load_prepared_case, load_from_manifest
from evaluation import score_episode

_DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject-id", help="Subject ID (use with --hadm-id)")
    group.add_argument("--manifest",   help="Manifest JSONL to pick a case from")
    parser.add_argument("--hadm-id",   help="Admission ID (use with --subject-id)")
    parser.add_argument("--index",     type=int, default=0, help="Case index in manifest (default: 0)")
    parser.add_argument("--model",  default=None,       help="OpenAI model override")
    parser.add_argument("--mode",   default="direct",   choices=["direct", "interactive"],
                        help="Environment mode (default: direct)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--hint-counts", action="store_true", help="Enable submission count hints")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.subject_id:
        if not args.hadm_id:
            print("--hadm-id required with --subject-id", file=sys.stderr)
            return 1
        case = load_prepared_case(args.subject_id, args.hadm_id)
    else:
        case = load_from_manifest(Path(args.manifest), args.index)

    if verbose:
        print(f"Loaded case: subject={case['subject_id']} hadm={case['hadm_id']} stages={len(case['stages'])}")

    episode_log = run_episode(case, model=args.model, mode=args.mode, verbose=verbose, hint_counts=args.hint_counts)
    scored      = score_episode(episode_log)

    sid   = case["subject_id"]
    hid   = case["hadm_id"]
    mode  = args.mode
    model = scored["model"]

    # Sanitize model name for use in filename (replace / and : with -)
    model_slug = model.replace("/", "-").replace(":", "-")

    target_dir = _DATA_DIR / "cases" / sid / model_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / f"{hid}_output.json"
    output = {
        "subject_id":    scored["subject_id"],
        "hadm_id":       scored["hadm_id"],
        "model":         model,
        "mode":          mode,
        "overall_score": scored["overall_score"],
        "overall_hits":  scored["overall_hits"],
        "overall_total": scored["overall_total"],
        "stages": [
            {
                "label":       s["label"],
                "index_range": s["index_range"],
                "gt":          s["gt"],
                "submissions": s.get("submissions", []),
                "score":       s.get("score", {}),
            }
            for s in scored["stages"]
        ],
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Episode complete.")
    print(f"Overall score: {scored['overall_hits']}/{scored['overall_total']} (F1={scored['overall_score']})")
    print(f"Output:   {output_path}")

    # Dialogue only written for interactive mode
    if mode == "interactive":
        dialogue_path = target_dir / f"{hid}_dialogue.json"
        dialogue = {
            "subject_id": scored["subject_id"],
            "hadm_id":    scored["hadm_id"],
            "model":      model,
            "mode":       mode,
            "stages": [
                {"label": s["label"], "index_range": s["index_range"], "messages": s.get("messages", [])}
                for s in scored["stages"]
            ],
        }
        dialogue_path.write_text(json.dumps(dialogue, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Dialogue: {dialogue_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
