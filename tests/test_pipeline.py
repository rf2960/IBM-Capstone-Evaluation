"""Runnable smoke tests for the refactored core and model/reference boundary."""
from pathlib import Path
import json
import pytest
import pandas as pd

from evaluation.config import load_config
from evaluation.datasets import load_dataset
from evaluation.metrics import (
    METRIC_REGISTRY, aggregate_prediction_scores, register_metric_group,
    score_prediction,
)
from evaluation.predictors import BASELINE_PREDICTORS, BaselinePredictor
from evaluation.analysis import temporary_reward_audit, temporary_reward_cases
from evaluation.protocol import metric_claim_specification
from evaluation.runner import (
    generate_predictions, evaluate_predictors_detailed,
    evaluate_external_predictions, run_experiment,
)
from evaluation.tasks import build_tasks

ROOT = Path(__file__).resolve().parents[1]


def sessions():
    return load_dataset({'type': 'curated', 'path': str(ROOT / 'examples/curated_sessions.jsonl')})


def test_config_baselines_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config(ROOT / 'configs/baseline.yaml')
    from dataclasses import replace
    cfg = replace(cfg, output=replace(cfg.output, directory=str(tmp_path)))
    predictions, detailed, summary = run_experiment(cfg)
    assert len(predictions) == len(detailed) == 20
    assert (tmp_path / 'summary.csv').exists()
    assert (tmp_path / 'predictions.csv').exists()
    assert (tmp_path / 'effective_config.json').exists()
    assert set(summary.baseline) == {'no_change', 'repeat_last_edit'}


def test_reference_information_does_not_reach_model():
    def spy(request):
        assert 'target' not in request and 'future' not in request
        assert 'outcome_timestamp' not in request
        assert 'history' in request
        return request['current']
    result = evaluate_predictors_detailed(sessions(), (1,), 'curated', {'spy': spy})
    assert (result['generation_status'] == 'ok').all()


def test_missing_prediction_is_censored_from_model_scores():
    s = sessions()
    original = generate_predictions(s, (1,), {'no_change': BaselinePredictor('no_change')})
    truncated = original.iloc[:-1].copy()
    details, summary = evaluate_external_predictions(s, truncated, cohort='curated')
    assert len(details) == len(original)
    assert details.generation_status.eq('missing').sum() == 1
    assert summary.invalid_predictions.iloc[0] == 1


def test_subset_metric_groups_only_emit_selected_scores():
    detail = evaluate_predictors_detailed(sessions(), (1,), 'curated',
                                           BASELINE_PREDICTORS,
                                           metric_groups=['immediate'])
    assert 'exact_match' in detail.columns
    assert 'stable_f1' not in detail.columns
    summary = aggregate_prediction_scores(detail, eligible_sessions=2,
                                          metric_groups=['immediate'])
    assert 'location_content_edit_f1' in summary.columns
    assert 'false_intervention_rate' not in summary.columns


def test_metric_registration_extension():
    name = '_test_custom_metric'
    try:
        register_metric_group(name, score=lambda shared: {'test_metric': 1.0},
                              aggregate=lambda group, scored: {'test_metric': float(scored['test_metric'].mean())})
        detailed = evaluate_predictors_detailed(sessions(), (1,), 'curated',
                                               BASELINE_PREDICTORS, metric_groups=[name])
        summary = aggregate_prediction_scores(detailed, eligible_sessions=2,
                                              metric_groups=[name])
        assert summary.test_metric.tolist() == [1.0, 1.0]
    finally:
        METRIC_REGISTRY.pop(name, None)


def test_invalid_curated_records_raise(tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text(json.dumps({'session_id': 'bad', 'states': ['x', None, 'y']})+'\n')
    with pytest.raises(ValueError, match='states must all be strings'):
        load_dataset({'type': 'curated', 'path': str(path)})


def test_file_predictor_roundtrip(tmp_path):
    from evaluation.predictors import FilePredictor
    records = generate_predictions(sessions(), (1,), {'no_change': BaselinePredictor('no_change')})
    path = tmp_path / 'predictions.csv'
    records.to_csv(path, index=False)
    replay = generate_predictions(sessions(), (1,), {
        'rescore': FilePredictor('rescore', path, source_model_name='no_change')
    })
    assert replay.prediction.tolist() == records.prediction.tolist()
    assert replay.generation_status.eq('ok').all()


def test_invalid_predictor_output_has_persistent_failure_status():
    records = generate_predictions(sessions(), (1,), {'bad': lambda request: None})
    assert records.generation_status.eq('invalid_output').all()
    assert records.prediction.isna().all()


def test_json_configuration_supported(tmp_path):
    config = {
        'experiment': 'json_config_test',
        'dataset': {'type': 'curated', 'path': str(ROOT / 'examples/curated_sessions.jsonl')},
        'models': [{'type': 'baseline', 'name': 'no_change'}],
        'evaluation': {'horizons': [1], 'metrics': ['immediate']},
    }
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    assert load_config(path).evaluation.metrics == ('immediate',)


def test_metric_claim_specification_covers_trajectory_aware_decisions():
    specification = metric_claim_specification().set_index('metric')
    required = {
        'exact_match', 'exact_match_within_future_window',
        'location_content_edit_f1', 'false_intervention_rate',
        'pooled_rework_reduction', 'mean_durability',
        'reference_reversion_rate', 'stable_f1', 'temporary_credit_share',
    }
    assert required.issubset(specification.index)
    assert not specification.isna().any().any()
    assert specification['denominator'].str.len().gt(0).all()
    assert specification['requires_additional_evidence'].str.len().gt(0).all()


def test_temporary_reward_audit_separates_immediate_and_stable_credit():
    source = [
        {'session_id': 'stable',
         'states': ['old\n', 'base\n', 'base\nx\n', 'base\nx\n']},
        {'session_id': 'temporary',
         'states': ['old\n', 'base\n', 'base\nx\n', 'base\n']},
    ]
    tasks = [task for task in build_tasks(source, 1)]
    detailed = pd.DataFrame([
        score_prediction(
            task, task['target'], cohort='test', model_name='copy_target'
        )
        for task in tasks
    ])
    audit = temporary_reward_audit(detailed, cohort='test', horizon=1).iloc[0]
    assert audit['persistence_eligible_tasks'] == 2
    assert audit['immediate_matched_credit'] == 2.0
    assert audit['stable_matched_credit'] == 1.0
    assert audit['temporary_credit_share'] == 0.5
    assert audit['exact_matches_with_only_transient_reference'] == 1

    cases = temporary_reward_cases(
        source, detailed, model_name='copy_target',
        predictor=lambda task: task['target'], cohort='test', limit=5,
    )
    assert len(cases) == 1
    assert bool(cases.iloc[0]['immediate_exact_match'])
    assert cases.iloc[0]['stable_matched_credit'] == 0.0
