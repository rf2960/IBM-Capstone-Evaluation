"""Trajectory-aware next-edit evaluation (refactored from evaluation_core.py)."""
from .config import load_config, EvaluationConfig
from .datasets import load_dataset, load_curated
from .metrics import METRIC_REGISTRY, register_metric_group, score_prediction, aggregate_prediction_scores
from .runner import evaluate_external_predictions, evaluate_predictors_detailed, generate_predictions, run_experiment
from .schemas import Session, EvalTask, PredictionRequest, PredictionRecord, Edit
from .analysis import temporary_reward_audit, temporary_reward_cases
from .protocol import metric_claim_specification, protocol_specification

__all__ = [
    'load_config', 'EvaluationConfig', 'load_dataset', 'load_curated',
    'METRIC_REGISTRY', 'register_metric_group', 'score_prediction',
    'aggregate_prediction_scores', 'evaluate_external_predictions',
    'evaluate_predictors_detailed', 'generate_predictions', 'run_experiment',
    'temporary_reward_audit', 'temporary_reward_cases',
    'metric_claim_specification', 'protocol_specification',
    'Session', 'EvalTask', 'PredictionRequest', 'PredictionRecord', 'Edit',
]
