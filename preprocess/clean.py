#!/usr/bin/env python3
"""Filter admission timelines for environment-ready preprocessing.

Selection rule (default):
- required_all:
  hosp_admissions_df,
  hosp_diagnoses_icd_df,
  hosp_labevents_df,
  note_df,
  hosp_procedures_icd_df
- required_any:
  hosp_prescriptions_df OR hosp_pharmacy_df
- required note headings:
  Chief Complaint,
  History of Present Illness/HPI,
  Brief Hospital Course,
  Discharge Diagnosis

Outputs:
- JSONL manifest (one selected admission per line)
- JSON summary
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_DATA_DIR = (
    "/Users/ylin766/Desktop/Benchmark/"
    "mimic-ext-time-series/Merge/ehr_by_subject"
)
DEFAULT_OUTPUT_DIR = "/Users/ylin766/Desktop/Benchmark/clean/data"

# NOTE: radiology_note intentionally removed from required_all.
DEFAULT_REQUIRED_ALL = (
    "hosp_admissions_df",
    "hosp_diagnoses_icd_df",
    "hosp_labevents_df",
    "note_df",
    "hosp_procedures_icd_df",
)
DEFAULT_REQUIRED_ANY = (
    "hosp_prescriptions_df",
    "hosp_pharmacy_df",
)

DEFAULT_REQUIRED_NOTE_SECTIONS = (
    "chief_complaint",
    "hpi",
    "brief_hospital_course",
    "discharge_diagnosis",
)

NOTE_SECTION_PATTERNS = {
    "chief_complaint": re.compile(r"(?im)^\s*Chief Complaint:\s*$"),
    "hpi": re.compile(r"(?im)^\s*(History of Present Illness|HPI):\s*$"),
    "brief_hospital_course": re.compile(r"(?im)^\s*Brief Hospital Course:\s*$"),
    "discharge_diagnosis": re.compile(r"(?im)^\s*Discharge Diagnosis:\s*$"),
}


@dataclass
class SelectedAdmission:
    subject_id: str
    hadm_id: str
    path: str
    tables: list[str]
    event_count: int
    required_note_sections: list[str]


def _iter_admission_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.glob("*/*.json")):
        if path.is_file():
            yield path


def _load_events(path: Path) -> list[dict]:
    events = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("Admission JSON must contain a list of events.")
    return events


def _tables_for_events(events: list[dict]) -> set[str]:
    return {event.get("source_table", "") for event in events if event.get("source_table")}


def _note_text_for_events(events: list[dict]) -> str:
    note_texts = [
        event.get("text", "")
        for event in events
        if event.get("source_table") == "note_df" and event.get("text")
    ]
    if not note_texts:
        return ""
    return max(note_texts, key=len)


def _has_required_note_sections(note_text: str, required_sections: tuple[str, ...]) -> bool:
    for section in required_sections:
        pattern = NOTE_SECTION_PATTERNS.get(section)
        if pattern is None:
            raise ValueError(f"Unknown required note section: {section}")
        if not pattern.search(note_text):
            return False
    return True


def build_manifest(
    data_dir: Path,
    required_all: tuple[str, ...],
    required_any: tuple[str, ...],
    required_note_sections: tuple[str, ...],
) -> tuple[list[SelectedAdmission], dict[str, int]]:
    selected: list[SelectedAdmission] = []
    stats = {
        "files_seen": 0,
        "files_zero_bytes": 0,
        "files_invalid_json": 0,
        "valid_admissions": 0,
        "missing_required_all": 0,
        "missing_required_any": 0,
        "missing_required_note_sections": 0,
        "selected_admissions": 0,
    }

    for path in _iter_admission_files(data_dir):
        stats["files_seen"] += 1

        if path.stat().st_size == 0:
            stats["files_zero_bytes"] += 1
            continue

        try:
            events = _load_events(path)
        except (json.JSONDecodeError, ValueError):
            stats["files_invalid_json"] += 1
            continue

        stats["valid_admissions"] += 1
        tables = _tables_for_events(events)

        if not set(required_all).issubset(tables):
            stats["missing_required_all"] += 1
            continue
        if required_any and not any(table in tables for table in required_any):
            stats["missing_required_any"] += 1
            continue

        if required_note_sections:
            note_text = _note_text_for_events(events)
            if not _has_required_note_sections(note_text, required_note_sections):
                stats["missing_required_note_sections"] += 1
                continue

        selected.append(
            SelectedAdmission(
                subject_id=path.parent.name,
                hadm_id=path.stem,
                path=str(path),
                tables=sorted(tables),
                event_count=len(events),
                required_note_sections=list(required_note_sections),
            )
        )

    stats["selected_admissions"] = len(selected)
    return selected, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--required-all",
        nargs="*",
        default=list(DEFAULT_REQUIRED_ALL),
        help="Tables that must all exist in an admission.",
    )
    parser.add_argument(
        "--required-any",
        nargs="*",
        default=list(DEFAULT_REQUIRED_ANY),
        help="At least one of these tables must exist in an admission.",
    )
    parser.add_argument(
        "--required-note-sections",
        nargs="*",
        default=list(DEFAULT_REQUIRED_NOTE_SECTIONS),
        choices=sorted(NOTE_SECTION_PATTERNS),
        help="Note sections that must appear as headings in note_df.text.",
    )
    parser.add_argument(
        "--prefix",
        default="env_ready_admissions",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected, stats = build_manifest(
        data_dir=data_dir,
        required_all=tuple(args.required_all),
        required_any=tuple(args.required_any),
        required_note_sections=tuple(args.required_note_sections),
    )

    manifest_path = output_dir / f"{args.prefix}.jsonl"
    summary_path = output_dir / f"{args.prefix}_summary.json"

    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    summary = {
        "data_dir": str(data_dir),
        "required_all": list(args.required_all),
        "required_any": list(args.required_any),
        "required_note_sections": list(args.required_note_sections),
        **stats,
        "selected_ratio": (
            round(stats["selected_admissions"] / stats["valid_admissions"], 4)
            if stats["valid_admissions"]
            else 0.0
        ),
        "manifest_path": str(manifest_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
