"""Prepared case writer.
Serializes a validated prepared case to disk and maintains a manifest index.
"""

import json
import os
from pathlib import Path

from preprocess.schemas.prepared_case_schema import validate_prepared_case

_DEFAULT_OUT_DIR = Path(__file__).parent.parent.parent / "data" / "cases"


def write_prepared_case(payload: dict, out_dir: Path = _DEFAULT_OUT_DIR) -> Path:
    """
    Validate and write a prepared case to disk.

    File layout:
        data/cases/<subject_id>/<hadm_id>.json

    Args:
        payload:  validated prepared case dict (subject_id, hadm_id, stages)
        out_dir:  root output directory (default: data/cases/)

    Returns:
        Path to the written file.

    Raises:
        ValueError if schema validation fails.
    """
    errors = validate_prepared_case(payload)
    if errors:
        raise ValueError("Prepared case failed validation:\n" + "\n".join(errors))

    subject_id = payload["subject_id"]
    hadm_id    = payload["hadm_id"]

    case_dir = out_dir / subject_id
    case_dir.mkdir(parents=True, exist_ok=True)

    out_path = case_dir / f"{hadm_id}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


def update_manifest(payload: dict, out_dir: Path = _DEFAULT_OUT_DIR) -> None:
    """
    Append or update an entry in data/cases/manifest.jsonl.
    Each line: {subject_id, hadm_id, path, n_stages}
    """
    manifest_path = out_dir / "manifest.jsonl"
    entry = {
        "subject_id": payload["subject_id"],
        "hadm_id":    payload["hadm_id"],
        "n_stages":   len(payload["stages"]),
        "path":       str(out_dir / payload["subject_id"] / f"{payload['hadm_id']}.json"),
    }

    # read existing entries, replace if same hadm_id exists
    entries = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                e = json.loads(line)
                entries[e["hadm_id"]] = e

    entries[entry["hadm_id"]] = entry

    manifest_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries.values()) + "\n",
        encoding="utf-8",
    )
