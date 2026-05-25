"""Per-case pipeline runner: interact → eval, with concurrency.

Each case is evaluated immediately after its dialogue completes.
Multiple cases run in parallel within a shard.
Supports checkpoint/resume: already-done cases are skipped.

Usage (one terminal per shard/port):
  python3 run_pipeline.py \
      --runtime  config/runtime-oss120b-port8000.json \
      --eval     config/eval-oss120b.json \
      --manifest data/manifest_393_shard0.jsonl \
      --workers  16

  # Dry-run to see what would run:
  python3 run_pipeline.py ... --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT       = Path(__file__).parent
CASES_DIR  = ROOT / "data" / "cases"
_print_lock = threading.Lock()


def _load_manifest(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(l) for l in lines if l.strip()]


def _model_slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def _has_run(sid: str, hid: str, slug: str) -> bool:
    return (CASES_DIR / sid / hid / "runs" / slug / "output.json").exists()


def _has_eval(sid: str, hid: str, slug: str) -> bool:
    return (CASES_DIR / sid / hid / "evals" / slug / "latest.json").exists()


def _run_cmd(cmd: list[str], env: dict) -> tuple[bool, str]:
    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT),
                       capture_output=True, text=True, env=env)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or "").strip()[-300:]


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _process_case(
    i: int, total: int, sid: str, hid: str,
    slug: str, model: str, do_process: bool, env: dict,
) -> tuple[str, str, str]:
    """Run interact → eval for one case. Returns (sid/hid, interact_status, eval_status)."""
    prefix = f"[{i:3d}/{total}] {sid}/{hid}"
    interact_status = eval_status = "skip"

    # ── Interact ────────────────────────────────────────────────────
    if not _has_run(sid, hid, slug):
        ok, err = _run_cmd(
            [sys.executable, "-m", "env.main",
             "--subject-id", sid, "--hadm-id", hid,
             "--model", model, "--quiet"],
            env,
        )
        if ok:
            interact_status = "ok"
            _log(f"{prefix}  interact=OK")
        else:
            interact_status = "fail"
            _log(f"{prefix}  interact=FAIL  {err}")
            return sid + "/" + hid, interact_status, "skip"
    else:
        _log(f"{prefix}  interact=SKIP(exists)")

    # ── Eval ────────────────────────────────────────────────────────
    if not _has_eval(sid, hid, slug):
        eval_cmd = [sys.executable, "-m", "evaluation.main",
                    "--subject-id", sid, "--hadm-id", hid,
                    f"--model={slug}", "--quiet"]
        if do_process:
            eval_cmd.append("--process")
        ok, err = _run_cmd(eval_cmd, env)
        if ok:
            eval_status = "ok"
            _log(f"{prefix}  eval=OK")
        else:
            eval_status = "fail"
            _log(f"{prefix}  eval=FAIL  {err}")
    else:
        _log(f"{prefix}  eval=SKIP(exists)")

    return sid + "/" + hid, interact_status, eval_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime",  required=True, help="Runtime config json path")
    parser.add_argument("--eval",     required=True, help="Eval config json path")
    parser.add_argument("--manifest", required=True, help="Manifest jsonl path")
    parser.add_argument("--workers",  type=int, default=8,
                        help="Concurrent cases per shard (default: 8)")
    parser.add_argument("--dry-run",  action="store_true", help="Print plan, don't execute")
    args = parser.parse_args()

    runtime_cfg = json.loads(Path(args.runtime).read_text())
    eval_cfg    = json.loads(Path(args.eval).read_text())
    model       = runtime_cfg.get("mut_model", "unknown")
    slug        = _model_slug(model)

    proc_conf  = eval_cfg.get("process_scorers", {})
    do_process = (proc_conf.get("info_coverage", {}).get("enabled", False) or
                  proc_conf.get("lab_cost",      {}).get("enabled", False))

    cases = _load_manifest(args.manifest)

    env = os.environ.copy()
    env["BENCHMARK_RUNTIME_CONFIG"] = str(Path(args.runtime).resolve())
    env["BENCHMARK_EVAL_CONFIG"]    = str(Path(args.eval).resolve())

    total      = len(cases)
    n_done     = sum(1 for c in cases if _has_eval(str(c["subject_id"]), str(c["hadm_id"]), slug))
    n_run_only = sum(1 for c in cases
                     if _has_run(str(c["subject_id"]), str(c["hadm_id"]), slug)
                     and not _has_eval(str(c["subject_id"]), str(c["hadm_id"]), slug))
    n_todo     = total - n_done - n_run_only

    print(f"[pipeline] model={model}", flush=True)
    print(f"[pipeline] manifest={args.manifest}  total={total}  "
          f"eval_done={n_done}  need_eval_only={n_run_only}  need_both={n_todo}  "
          f"workers={args.workers}", flush=True)

    if args.dry_run:
        return

    interact_ok = interact_fail = eval_ok = eval_fail = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _process_case,
                i, total,
                str(c["subject_id"]), str(c["hadm_id"]),
                slug, model, do_process, env,
            ): c
            for i, c in enumerate(cases, 1)
        }
        for fut in concurrent.futures.as_completed(futures):
            _, i_status, e_status = fut.result()
            if i_status == "ok":   interact_ok   += 1
            if i_status == "fail": interact_fail += 1
            if e_status == "ok":   eval_ok        += 1
            if e_status == "fail": eval_fail      += 1

    print(f"\n[pipeline] DONE  interact ok={interact_ok} fail={interact_fail}  "
          f"eval ok={eval_ok} fail={eval_fail}", flush=True)


if __name__ == "__main__":
    main()
