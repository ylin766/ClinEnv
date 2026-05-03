"""Unified benchmark runner.

Stages:
  plan      — preprocess EHR → plan stages → build readviews → write case/case.json
  interact  — run model episodes → write runs/<model>/output.json + dialogue.json
  eval      — score episodes → write evals/<model>/latest.json
  all       — run plan → interact → eval in sequence

Config (read automatically, no flags needed for normal use):
  config/run.json      — manifest, limit, concurrency, skip_existing
  config/planner.json  — planner model (used by plan stage)
  config/runtime.json  — MUT model, env model, interaction mode
  config/eval.json     — scoring toggles (process scorers, etc.)

Usage:
  python3 run.py plan
  python3 run.py interact
  python3 run.py eval
  python3 run.py all
  python3 run.py plan --manifest data/manifest_batch_new.jsonl --limit 5
  python3 run.py interact --limit 20 --concurrency 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import concurrent.futures
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

_ROOT      = Path(__file__).parent
_CASES_DIR = _ROOT / "data" / "cases"


# ------------------------------------------------------------------ #
# Config helpers                                                       #
# ------------------------------------------------------------------ #

_runtime_config_path: Path | None = None  # overridden by --runtime-config


def _cfg(name: str) -> dict:
    if name == "runtime" and _runtime_config_path is not None:
        p = _runtime_config_path
    else:
        p = _ROOT / "config" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _mut_model() -> str:
    return _cfg("runtime").get("mut_model", "gpt-5.4")


def _model_slug(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


# ------------------------------------------------------------------ #
# Existence checks — one per stage                                     #
# ------------------------------------------------------------------ #

def _has_case(sid: str, hid: str) -> bool:
    return (_CASES_DIR / sid / hid / "case" / "case.json").exists()


def _has_run(sid: str, hid: str, slug: str) -> bool:
    return (_CASES_DIR / sid / hid / "runs" / slug / "output.json").exists()


def _has_eval(sid: str, hid: str, slug: str) -> bool:
    return (_CASES_DIR / sid / hid / "evals" / slug / "latest.json").exists()


# ------------------------------------------------------------------ #
# Case list helpers                                                    #
# ------------------------------------------------------------------ #

def _load_manifest(path: str) -> list[dict]:
    p = _ROOT / path
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _select(cases: list[dict], skip_fn, limit: int | None) -> tuple[list[dict], int, int]:
    """Filter cases that need processing, then apply limit.

    Returns (todo, already_done, deferred) where:
      already_done — cases that have existing output (skip_existing=true)
      deferred     — cases that need processing but exceed the limit
    """
    pending      = [c for c in cases if not skip_fn(str(c["subject_id"]), str(c["hadm_id"]))]
    already_done = len(cases) - len(pending)
    todo         = pending[:limit] if limit else pending
    deferred     = len(pending) - len(todo)
    return todo, already_done, deferred


# ------------------------------------------------------------------ #
# Stage runners                                                        #
# ------------------------------------------------------------------ #

def _run_cmd(cmd: list[str]) -> tuple[bool, str]:
    import os
    env = os.environ.copy()
    if _runtime_config_path is not None:
        env["BENCHMARK_RUNTIME_CONFIG"] = str(_runtime_config_path)
    try:
        subprocess.run(cmd, check=True, cwd=str(_ROOT),
                       capture_output=True, text=True, env=env)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, (e.stderr or "").strip()[-200:]


def stage_plan(conf: dict, manifest: str | None, limit: int | None,
               concurrency: int | None, verbose: bool) -> None:
    run_conf  = _cfg("run")
    manifest  = manifest  or run_conf.get("manifest", "data/batch_500_manifest.jsonl")
    limit     = limit     if limit is not None else run_conf.get("limit")
    skip      = run_conf.get("skip_existing", True)
    workers   = concurrency or run_conf.get("concurrency", {}).get("plan", 4)

    all_cases            = _load_manifest(manifest)
    todo, done, deferred = _select(all_cases, (lambda s, h: skip and _has_case(s, h)), limit)
    print(f"[plan]  total={len(all_cases)}  done={done}  run={len(todo)}  deferred={deferred}  workers={workers}")
    if not todo:
        return

    def _one(c: dict) -> str:
        sid, hid = str(c["subject_id"]), str(c["hadm_id"])
        ok, err  = _run_cmd([sys.executable, "-m", "preprocess.main",
                              "--subject-id", sid, "--hadm-id", hid,
                              "--reuse-plan", "--quiet"])
        return f"{'OK' if ok else 'FAIL'} {sid}/{hid}" + (f" — {err}" if err else "")

    done = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, result in enumerate(ex.map(_one, todo), 1):
            ok = result.startswith("OK")
            done += ok; fail += not ok
            if verbose or not ok:
                print(f"  [{i}/{len(todo)}] {result}", flush=True)
    print(f"[plan]  done={done}  fail={fail}")


def stage_interact(conf: dict, manifest: str | None, limit: int | None,
                   concurrency: int | None, verbose: bool) -> None:
    run_conf  = _cfg("run")
    manifest  = manifest  or run_conf.get("manifest", "data/batch_500_manifest.jsonl")
    limit     = limit     if limit is not None else run_conf.get("limit")
    skip      = run_conf.get("skip_existing", True)
    workers   = concurrency or run_conf.get("concurrency", {}).get("interact", 2)
    model     = _mut_model()
    slug      = _model_slug(model)

    all_cases            = _load_manifest(manifest)
    planned              = [c for c in all_cases if _has_case(str(c["subject_id"]), str(c["hadm_id"]))]
    no_plan              = len(all_cases) - len(planned)
    todo, done, deferred = _select(planned, (lambda s, h: skip and _has_run(s, h, slug)), limit)
    print(f"[interact]  model={model}  total={len(all_cases)}  no_plan={no_plan}  "
          f"done={done}  run={len(todo)}  deferred={deferred}  workers={workers}")
    if not todo:
        return

    def _one(c: dict) -> str:
        sid, hid = str(c["subject_id"]), str(c["hadm_id"])
        ok, err  = _run_cmd([sys.executable, "-m", "env.main",
                              "--subject-id", sid, "--hadm-id", hid,
                              "--model", model, "--quiet"])
        return f"{'OK' if ok else 'FAIL'} {sid}/{hid}" + (f" — {err}" if err else "")

    done = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, result in enumerate(ex.map(_one, todo), 1):
            ok = result.startswith("OK")
            done += ok; fail += not ok
            if verbose or not ok:
                print(f"  [{i}/{len(todo)}] {result}", flush=True)
    print(f"[interact]  done={done}  fail={fail}")


def stage_eval(conf: dict, manifest: str | None, limit: int | None,
               concurrency: int | None, verbose: bool) -> None:
    run_conf  = _cfg("run")
    eval_conf = _cfg("eval")
    manifest  = manifest  or run_conf.get("manifest", "data/batch_500_manifest.jsonl")
    limit     = limit     if limit is not None else run_conf.get("limit")
    skip      = run_conf.get("skip_existing", True)
    workers   = concurrency or run_conf.get("concurrency", {}).get("eval", 4)
    model     = _mut_model()
    slug      = _model_slug(model)

    proc_conf   = eval_conf.get("process_scorers", {})
    do_process  = (proc_conf.get("info_coverage", {}).get("enabled", False) or
                   proc_conf.get("lab_cost",      {}).get("enabled", False))

    all_cases            = _load_manifest(manifest)
    has_run              = [c for c in all_cases if _has_run(str(c["subject_id"]), str(c["hadm_id"]), slug)]
    no_run               = len(all_cases) - len(has_run)
    todo, done, deferred = _select(has_run, (lambda s, h: skip and _has_eval(s, h, slug)), limit)
    print(f"[eval]  model={model}  total={len(all_cases)}  no_run={no_run}  "
          f"done={done}  run={len(todo)}  deferred={deferred}  workers={workers}")
    if not todo:
        return

    def _one(c: dict) -> str:
        sid, hid = str(c["subject_id"]), str(c["hadm_id"])
        cmd = [sys.executable, "-m", "evaluation.main",
               "--subject-id", sid, "--hadm-id", hid,
               "--model", slug, "--quiet"]
        if do_process:
            cmd.append("--process")
        ok, err = _run_cmd(cmd)
        return f"{'OK' if ok else 'FAIL'} {sid}/{hid}" + (f" — {err}" if err else "")

    done = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, result in enumerate(ex.map(_one, todo), 1):
            ok = result.startswith("OK")
            done += ok; fail += not ok
            if verbose or not ok:
                print(f"  [{i}/{len(todo)}] {result}", flush=True)
    print(f"[eval]  done={done}  fail={fail}")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=["plan", "interact", "eval", "all"],
                        help="Which pipeline stage to run")
    parser.add_argument("--manifest",       help="Override config manifest path")
    parser.add_argument("--limit",          type=int, help="Override config limit")
    parser.add_argument("--concurrency",    type=int, help="Override config concurrency for this stage")
    parser.add_argument("--runtime-config", help="Alternative runtime.json path")
    parser.add_argument("--quiet",          action="store_true")
    args = parser.parse_args()

    global _runtime_config_path
    if args.runtime_config:
        _runtime_config_path = Path(args.runtime_config)

    verbose = not args.quiet
    conf    = _cfg("run")
    kwargs  = dict(conf=conf, manifest=args.manifest,
                   limit=args.limit, concurrency=args.concurrency, verbose=verbose)

    if args.stage in ("plan", "all"):
        stage_plan(**kwargs)
    if args.stage in ("interact", "all"):
        stage_interact(**kwargs)
    if args.stage in ("eval", "all"):
        stage_eval(**kwargs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
