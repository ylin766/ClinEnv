# ClinEnv Dataset

ClinEnv is a clinical decision-making benchmark derived from MIMIC-IV. Each case represents a real hospital admission structured as a sequence of evaluation stages. A model under test (MUT) is given access to a partial EHR context and must identify the clinical decisions made at each stage.

## Dataset Structure

```
ClinEnv_dataset/
  README.md                               # this file
  manifest.jsonl                          # index of all cases
  {subject_id}/
    {hadm_id}/
      case/
        case.json                         # prepared case file
```

### manifest.jsonl

One JSON object per line, each with `subject_id` and `hadm_id` identifying a case.

---

## case.json Schema

```json
{
  "subject_id": "string",
  "hadm_id": "string",
  "stages": [ ... ],
  "prior_admissions": [ ... ]
}
```

### stages

Each stage represents a decision point in the admission. Stages are ordered chronologically and non-overlapping.

```json
{
  "context_range": [int, int],
  "gts": [ ... ],
  "events": [ ... ],
  "visible_events": [ ... ],
  "available_agents": ["patient", "nurse", "lab", "history"],
  "readviews": {
    "patient": { ... },
    "nurse":   { ... },
    "lab":     { ... },
    "history": { ... }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `context_range` | `[int, int]` | Inclusive index range of EHR events visible to the MUT at this stage |
| `gts` | list | Ground-truth clinical decisions made at this stage (see below) |
| `events` | list | All EHR timeline events in the context window |
| `visible_events` | list | Subset of events surfaced to the MUT (pre-filtered) |
| `available_agents` | list | Which information agents are active at this stage |
| `readviews` | dict | Agent-specific filtered views of the EHR (see Agent Readviews) |

### gts (ground-truth items)

Each item in `gts` is one clinical decision. Three types are supported:

**medication**
```json
{
  "type": "medication",
  "description": "string",
  "action": "start | stop | adjust | switch",
  "event_index": int,
  "drug_name": "string",
  "dose": "string",
  "dose_unit": "string",
  "route": "string",
  "frequency": "string",
  "prod_strength": "string"
}
```

**procedure**
```json
{
  "type": "procedure",
  "description": "string",
  "event_index": int,
  "icd_code": "string",
  "icd_version": 9 | 10,
  "procedure_title": "string"
}
```

**diagnosis**
```json
{
  "type": "diagnosis",
  "description": "string",
  "icd_code": "string",
  "icd_version": 9 | 10,
  "diagnosis_title": "string",
  "primary": true | false
}
```

`event_index` is absent for plan-level decisions (no single anchoring EHR event).  
`unscoreable: true` may appear on items that could not be enriched with structured fields; these are excluded from scoring.

### Agent Readviews

Each readview is a filtered projection of the EHR restricted to what that agent role can see:

| Agent | Content |
|---|---|
| `patient` | Demographics, chief complaint, HPI, past medical history |
| `nurse` | Vitals, fluid balance, medication administration records |
| `lab` | Laboratory and microbiology results |
| `history` | Prior discharge summaries (only present if prior admissions exist) |

### prior_admissions

List of prior admission summaries for the patient, used by the history agent. May be empty.

---

## Evaluation Modes

Cases support two evaluation modes:

| Mode | Description |
|---|---|
| `direct` | MUT receives the full context window and submits decisions directly |
| `interactive` | MUT queries information agents before submitting decisions |

---

## Data Source

Cases are derived from [MIMIC-IV](https://physionet.org/content/mimiciv/), a de-identified EHR database. Access requires credentialed PhysioNet registration and completion of CITI training. This dataset does not redistribute raw MIMIC-IV data; all fields are derived summaries and structured annotations.
