# Clinical Decision Benchmark

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY
```

---

## 1. Planner (preprocessing)

Generates a structured case plan from raw EHR data.

**Single case:**
```bash
python -m preprocess.main --subject-id 10511716 --hadm-id 24341177
```

**Batch from manifest:**
```bash
python -m preprocess.main --manifest data/env_ready_admissions_p50.jsonl
python -m preprocess.main --manifest data/env_ready_admissions_p50.jsonl --limit 10
```

Output: `data/cases/{subject_id}/{hadm_id}.json`

| Flag | Description |
|---|---|
| `--subject-id` | Subject ID (requires `--hadm-id`) |
| `--hadm-id` | Admission ID |
| `--manifest` | JSONL manifest for batch processing |
| `--limit N` | Max cases to process from manifest |
| `--quiet` | Suppress verbose output |

---

## 2. Environment (evaluation)

Runs a prepared case through the doctor model and scores the result.

**Single case:**
```bash
python -m env.main --subject-id 10511716 --hadm-id 24341177
python -m env.main --subject-id 10511716 --hadm-id 24341177 --mode interactive
```

**From manifest:**
```bash
python -m env.main --manifest data/cases/manifest.jsonl --index 0
python -m env.main --manifest data/cases/manifest.jsonl --index 0 --model gpt-5.4-mini-2026-03-17
```

Output:
- `data/dialogue/{subject_id}/{hadm_id}_{mode}.json` — full conversation log
- `data/output/{subject_id}/{hadm_id}_{mode}.json` — submissions + scores

| Flag | Description |
|---|---|
| `--subject-id` / `--hadm-id` | Single case |
| `--manifest` + `--index` | Case from manifest |
| `--mode` | `direct` (default) or `interactive` |
| `--model` | Model name override (default: `gpt-5.4-mini-2026-03-17`) |
| `--quiet` | Suppress verbose output |

---

## 3. Modes

| Mode | Description |
|---|---|
| `direct` | Model receives the full clinical chart and submits decisions directly |
| `interactive` | Model queries agents (patient, nurse, lab, history) before submitting |

---

## 4. Programmatic use

```python
from env.readers.prepared_case_reader import load_prepared_case
from env import run_episode
from evaluation import score_episode

case = load_prepared_case("10511716", "24341177")

# standard run
episode_log = run_episode(case, mode="direct")

# with submission count hints (experimental: tells model how many of each type to submit)
episode_log = run_episode(case, mode="direct", hint_counts=True)

scored = score_episode(episode_log)
print(scored["overall_score"])  # overall F1
```

---

## Data layout

```
data/
  cases/       # prepared case plans (planner output)
  dialogue/    # per-episode conversation logs
  output/      # per-episode submissions + scores
```
