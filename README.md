# Modular Next-Edit Prediction Evaluation

A minimal refactor of the supplied **`evaluation_core.py`** into a reusable, configurable trajectory-aware evaluation pipeline. Existing frozen metrics retain their definitions. This repository includes the original code at `legacy/evaluation_core_original.py` for reference and comparison.

## What was already good

The original evaluation had a useful and carefully defined foundation: line-diff/edit-block extraction; **one-to-one optimal matching** of location/content; separate immediate, false-intervention, rework, durability and **stable F1** metrics; explicit denominators and censored follow-up; a **frozen protocol**; and an external prediction-table contract. It also contained trajectory-level bootstrap, validity/case-review tools, stress cases, and descriptive breakdowns. These functions were mostly **moved, not redesigned**.

## Key changes

| File | Role |
|---|---|
| `evaluation/schemas.py` | `Session`, `EvalTask`, safe `PredictionRequest`, `PredictionRecord`, existing `Edit` |
| `datasets.py` / `tasks.py` | Load pre-constructed DECODE or curated trajectories; construct windows |
| `predictors.py` | Callable baselines, saved prediction files, local Hugging Face causal LM + optional LoRA adapter |
| `edits.py` | Original diff/matching/persistence primitives |
| `metrics.py` | Registry of scoring and **aggregation** function pairs: `immediate`, `intervention`, `rework`, `persistence` |
| `runner.py` / `cli.py` | Shared inference-to-record-to-scoring workflow, standalone and external predictions |
| `config.py` | Validated YAML or JSON experiment settings |
| `analysis.py` | Existing audits, reports, bootstrap and human-review helpers |
| `constants.py` / `protocol.py` | Existing frozen metric definitions and protocol |

**Important:** A model only receives `PredictionRequest` (observed history and current code), *not target or future states*. Legacy function predictors also receive a restricted dictionary. Predictions have a common output contract: **the full predicted code file**, not a patch. Metrics still use pooled block counts for micro-F1, not mean example-level F1.

## Quick start

From the extracted project root:

```bash
pip install -e '.[test]'
python -m evaluation.cli --config configs/baseline.yaml
python -m pytest -q
```

The included `examples/curated_sessions.jsonl` has two toy trajectories, so the baseline command runs immediately. Outputs are saved under `results/baseline_comparison/`: `predictions.csv` (reusable inference outputs), `prediction_level.csv`, `summary.csv`, `effective_config.json`, and `evaluation_protocol.json`.

For **preprocessed DECODE trajectories**, set `dataset.type: decode`, use a `.parquet` path (install `pip install -e '.[parquet]'`), and choose `cohort: strict_no_missing_states` or `drop_none_states`. Input must already have the original expected `states`, snapshot and audit columns; this package **does not** perform raw-data cleaning, event segmentation, or curation. For **curated trajectories**, use `type: curated` with JSONL, JSON, CSV (JSON-encoded `states` array), or Parquet. Each record needs `session_id` and an ordered `states` array containing at least three code strings. Optional aligned timestamps can be loaded via `timestamps_column`.

For a tuned model, see `configs/hf_lora.example.yaml` and install `pip install -e '.[models]'`. Use an accessible `model_id` and optional compatible PEFT `adapter_path`. The basic adapter performs deterministic generation and expects full-file output; **prompt formatting and parsing must be adapted to the chosen checkpoint**. GPU inference has not been tested in this deliverable. For previously generated outputs, configure a `type: file` predictor pointing to a CSV/JSONL/Parquet prediction table.

## Extending the system

Add a metric by implementing `score(shared) -> dict` and `aggregate(group, scored) -> dict`, then registering it with `register_metric_group()`. New groups can be selected in YAML once imported/registered. Add a dataset by returning normalized `Session` instances from another loader; add a model by implementing `predict(PredictionRequest) -> PredictionRecord`. No evaluator changes should be needed for ordinary extensions.

## Scope and verification

`tests/` includes runnable core, safe-input, missing-prediction, metric-registry, and original-vs-refactored **toy-data parity** checks. The archived original and refactor match across all original detailed and aggregate columns on the supplied example. This is **not** full real-DECODE validation.

`scaffolding/` contains **inactive interfaces/placeholders** for prediction cache, verified run manifest, centralized validity policy, dataset-namespaced IDs, and future/lead-time/survival metrics. The real-data golden-fixture test is explicitly skipped. None is represented as complete or enabled, and the original frozen metrics are not silently redefined.
