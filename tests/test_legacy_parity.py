"""Baseline comparison with retained original code; frozen metric semantics."""
import pandas as pd
import legacy.evaluation_core_original as legacy
from evaluation.datasets import load_dataset
from evaluation.predictors import BASELINE_PREDICTORS
from evaluation.runner import evaluate_predictors_detailed, evaluate_predictions
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'examples/curated_sessions.jsonl'


def test_frozen_detailed_and_aggregate_results_match_original():
    sessions = [s.to_legacy() for s in load_dataset({'type': 'curated', 'path': str(DATA)})]
    for horizons in [(1,), (1, 2)]:
        old_details = legacy.evaluate_predictors_detailed(sessions, horizons, 'curated', legacy.BASELINE_PREDICTORS)
        new_details = evaluate_predictors_detailed(sessions, horizons, 'curated', BASELINE_PREDICTORS)
        pd.testing.assert_frame_equal(old_details, new_details, check_like=True)
        old_summary = legacy.evaluate_predictions(sessions, horizons, 'curated', legacy.BASELINE_PREDICTORS)
        new_summary = evaluate_predictions(sessions, horizons, 'curated', BASELINE_PREDICTORS)
        # Column order is not semantic; values and columns must match exactly.
        pd.testing.assert_frame_equal(old_summary[sorted(old_summary.columns)],
                                      new_summary[sorted(new_summary.columns)])
