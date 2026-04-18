# Clinical Interaction Environment

An interactive clinical evaluation environment built on longitudinal MIMIC-style EHR data.
The environment places a doctor model inside a realistic clinical episode where it must
query agents, request tests, and submit decisions at structured decision stages.

## Overview

The system operates in two phases:

**Preprocessing (offline)**
Raw EHR timelines are loaded, analyzed by a planner LLM, and serialized into
*prepared cases*. Each prepared case contains:
- Structured decision stages with task descriptions and ground truth
- Per-stage, per-agent readviews (what each agent can see at that point in time)
- Patient history (prior admissions) in two layers: summary and full detail

**Runtime (online)**
The environment controller steps through a prepared case stage by stage.
At each stage, the doctor model may call tools to gather information, then submits a decision.

## Environment Components

| Component | Role |
|---|---|
| **Patient agent** | Answers questions about symptoms, complaints, and patient-reported history |
| **Nurse agent** | Answers questions about vitals, monitoring data, and bedside observations |
| **Lab agent** | Returns lab results when the doctor explicitly requests a specific test |
| **recall_history tool** | Provides prior admission records — summary (layer 1) or full detail (layer 2) |
| **submit_decision tool** | Receives and validates the doctor's decision at each stage |
| **calculator tool** | Performs deterministic clinical formula calculations |

## Directory Structure

```
preprocess/          # Offline pipeline: raw EHR → prepared cases
  loaders/           # Load raw admission events from MIMIC timeline JSON
  planners/          # LLM-based planner: generate decision stages + ground truth
  readviews/         # Build per-stage, per-agent data views
  schemas/           # Validate prepared case structure
  writers/           # Serialize prepared cases to disk

env/                 # Runtime environment
  agents/            # Patient, nurse, lab agent implementations
  tools/             # Doctor-callable tools (submit, recall_history, calculator)
  runtime/           # Environment controller: episode loop
  readers/           # Load prepared cases at runtime
  memory/            # Dialogue turn history within an episode
  logging/           # Persist episode traces

data/                # Processed case outputs
prompts/             # System prompts for agents and planner
results/             # Episode logs (gitignored)
```

## Data

Built on MIMIC-IV Extended timelines (`mimic-ext-time-series/Merge/ehr_by_subject/`).
After filtering for data completeness, **7,098 admissions** qualify as environment-ready.
Filtering criteria: all core EHR tables present, required discharge note sections present.

## Out of Scope

- Evaluation metrics and scoring
- Judge models
- Benchmark reporting

## TODO

### Preprocessing Pipeline
- [ ] `preprocess/schemas/prepared_case_schema.py` — define and validate prepared case structure
- [ ] `preprocess/loaders/ehr_loader.py` — load raw admission events by subject/hadm
- [ ] `preprocess/planners/decision_planner.py` — LLM planner: extract decision stages and ground truth
- [ ] `preprocess/readviews/readview_builder.py` — build per-stage readviews for each agent
- [ ] `preprocess/writers/prepared_case_writer.py` — serialize prepared cases, update manifest
- [ ] `preprocess/main.py` — orchestrate the full preprocessing pipeline

### Runtime Environment
- [ ] `env/readers/prepared_case_reader.py` — load prepared cases at runtime
- [ ] `env/agents/patient_agent.py` — LLM-backed patient Q&A
- [ ] `env/agents/nurse_agent.py` — LLM-backed nurse Q&A on vitals and monitoring
- [ ] `env/agents/lab_agent.py` — return lab results on explicit doctor request
- [ ] `env/tools/recall_history_tool.py` — two-layer prior admission access
- [ ] `env/tools/submit_decision_tool.py` — validate and record doctor decisions
- [ ] `env/tools/calculator_tool.py` — clinical formula calculations
- [ ] `env/memory/dialogue_memory.py` — record dialogue turns within an episode
- [ ] `env/runtime/environment_controller.py` — main episode loop
- [ ] `env/logging/episode_logger.py` — persist episode traces
- [ ] `env/main.py` — runtime entry point

### Prompts
- [ ] Agent system prompts (patient, nurse, lab)
- [ ] Planner system prompt
- [ ] Doctor model system prompt
