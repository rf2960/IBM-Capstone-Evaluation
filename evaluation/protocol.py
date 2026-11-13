from __future__ import annotations

from .constants import PROTOCOL_VERSION, PRIMARY_SCORECARD_COLUMNS, PREDICTION_REQUIRED_COLUMNS


def protocol_specification() -> dict:
    """Frozen evaluation choices established before real model comparison."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_on": "2026-11-04",
        "status": "frozen_before_real_model_results",
        "primary_population": "strict_no_missing_states",
        "primary_horizon": 1,
        "primary_scorecard_columns": list(PRIMARY_SCORECARD_COLUMNS),
        "metric_directions": {
            "prediction_coverage": "higher",
            "exact_match": "higher",
            "location_content_edit_f1": "higher",
            "false_intervention_rate": "lower",
            "pooled_rework_reduction": "higher",
            "stable_f1": "higher",
        },
        "edit_matching": "global_one_to_one_linear_assignment",
        "no_edit_definition": "no non-equal line block after trailing-whitespace normalization",
        "primary_persistence_min_consecutive_states": 2,
        "persistence_sensitivity_states": [3, 5],
        "censoring": "insufficient future observation is censored, not labeled unstable",
        "uncertainty": {
            "unit": "trajectory_cluster",
            "bootstrap_resamples": 2000,
            "seed": 2026,
        },
        "split": {
            "unit": "user_id",
            "seed": 20261104,
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
        },
        "prediction_required_columns": list(PREDICTION_REQUIRED_COLUMNS),
        "prediction_optional_audit_columns": ["model_version", "prompt_version"],
        "change_policy": "future changes must be versioned and labeled secondary unless a defect is demonstrated",
    }

