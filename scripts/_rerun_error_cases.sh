#!/bin/bash
# 清除 29 个 error 案例的 process_score，然后重跑 process 评估
# 用法: bash scripts/_rerun_error_cases.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
ERROR_FILE="$SCRIPT_DIR/_rerun_process_errors.json"

echo "清除 error 案例的 process_score..."
python3 - <<PYEOF
import json
from pathlib import Path

error_file = Path("$ERROR_FILE")
cases = json.loads(error_file.read_text())
cleared = 0
for c in cases:
    path = Path("data/cases") / c["subject_id"] / c["hadm_id"] / "evals" / "gpt-5.4" / "latest.json"
    if not path.exists():
        continue
    d = json.loads(path.read_text())
    for s in d.get("stages", []):
        if "process_score" in s:
            del s["process_score"]
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    cleared += 1
print(f"  已清除 process_score: {cleared} 个案例")
PYEOF

echo "重新运行 process 评估..."
python3 -m evaluation.main --all --model gpt-5.4 --process --workers 3 --skip-process-done

echo "完成！"
