"""Prepared case reader.
Loads preprocessed case data from data/cases/.
"""

import json
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "cases"


def load_prepared_case(subject_id: str, hadm_id: str) -> dict:
    path = _DATA_DIR / subject_id / hadm_id / "case" / "case.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_from_manifest(manifest_path: Path, index: int = 0) -> dict:
    entries = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    e = entries[index]
    # Support both old path format and new path from manifest
    if "path" in e:
        return json.loads(Path(e["path"]).read_text(encoding="utf-8"))
    return load_prepared_case(e["subject_id"], e["hadm_id"])
