import json
import argparse
import subprocess
import concurrent.futures
from pathlib import Path

def run_one_planner(subject_id, hadm_id, model=None):
    cmd = [
        "python3", "-m", "preprocess.main",
        "--subject-id", str(subject_id),
        "--hadm-id", str(hadm_id),
        "--reuse-plan",
        "--quiet"
    ]
    if model:
        cmd.extend(["--model", model])
    
    try:
        subprocess.run(cmd, check=True)
        return f"[SUCCESS] {subject_id}/{hadm_id}"
    except subprocess.CalledProcessError as e:
        return f"[FAILED] {subject_id}/{hadm_id}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    # Load config
    config_path = Path("config/planner.json")
    with open(config_path, "r") as f:
        conf = json.load(f)
    
    concurrency = conf.get("concurrency", 8)
    model = conf.get("model")

    # Load manifest
    with open(args.manifest, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Starting parallel planner for {len(records)} cases with concurrency={concurrency}...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_one_planner, r["subject_id"], r["hadm_id"], model) for r in records]
        
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            done_count += 1
            print(f"[{done_count}/{len(records)}] {res}", flush=True)

if __name__ == "__main__":
    main()
