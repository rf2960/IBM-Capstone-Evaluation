"""End-to-end inference/scoring; model adapters never receive future/target."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Iterable
import json
import pandas as pd

from .constants import PREDICTION_REQUIRED_COLUMNS
from .datasets import load_dataset
from .metrics import aggregate_prediction_scores, score_prediction, validate_metric_groups
from .predictors import BASELINE_PREDICTORS, Predictor, load_predictors
from .schemas import EvalTask, PredictionRecord, PredictionRequest
from .tasks import build_tasks
from .protocol import protocol_specification

PredictionFn = Callable[[dict], str]


def _invoke(predictor: Predictor | PredictionFn, request: PredictionRequest,
            model_name: str) -> PredictionRecord:
    """Legacy function predictors get only an input dict; adapters get request."""
    try:
        if hasattr(predictor, 'predict'):
            result = predictor.predict(request)
        else:
            result = predictor(request.as_input_dict())
        if isinstance(result, PredictionRecord):
            if (result.example_id != request.example_id or result.horizon != request.horizon
                    or result.model_name != model_name):
                raise ValueError('Predictor returned mismatched identifying fields')
            if result.generation_status == 'ok' and not isinstance(result.prediction, str):
                return replace(result, generation_status='invalid_output')
            return result
        return PredictionRecord(request.example_id, model_name, request.horizon, result,
                                generation_status=('ok' if isinstance(result, str)
                                                   else 'invalid_output'))
    except Exception as error:
        return PredictionRecord(request.example_id, model_name, request.horizon,
                                None, generation_status=f'error:{type(error).__name__}')


def generate_predictions(sessions, horizons: tuple[int, ...], predictors: dict) -> pd.DataFrame:
    """Generate exactly once per task/model; reusable with external scorer."""
    rows = []
    for horizon in horizons:
        for task in build_tasks(sessions, horizon):
            request = EvalTask.from_legacy(task).prediction_request()
            for name, predictor in predictors.items():
                rows.append(_invoke(predictor, request, name).as_dict())
    return pd.DataFrame(rows, columns=[*PREDICTION_REQUIRED_COLUMNS, 'model_version', 'prompt_version'])


def evaluate_predictors_detailed(
    sessions: list[dict], horizons: tuple[int, ...], cohort: str,
    predictors: dict[str, Predictor | PredictionFn], *,
    persistence_min_consecutive: int = 2,
    metric_groups: Iterable[str] | None = None,
) -> pd.DataFrame:
    rows = []
    names = validate_metric_groups(metric_groups)
    for horizon in horizons:
        for task in build_tasks(sessions, horizon):
            request = EvalTask.from_legacy(task).prediction_request()
            for model_name, predictor in predictors.items():
                record = _invoke(predictor, request, model_name)
                rows.append(score_prediction(
                    task, record.prediction, cohort=cohort, model_name=model_name,
                    generation_status=record.generation_status,
                    model_version=record.model_version, prompt_version=record.prompt_version,
                    persistence_min_consecutive=persistence_min_consecutive,
                    metric_groups=names,
                ))
    return pd.DataFrame(rows)


def evaluate_predictions(sessions: list[dict], horizons: tuple[int, ...], cohort: str,
                         predictors: dict, *, persistence_min_consecutive: int = 2,
                         metric_groups: Iterable[str] | None = None) -> pd.DataFrame:
    summaries = []
    for horizon in horizons:
        detailed = evaluate_predictors_detailed(
            sessions, (horizon,), cohort, predictors,
            persistence_min_consecutive=persistence_min_consecutive, metric_groups=metric_groups,
        )
        summaries.append(aggregate_prediction_scores(
            detailed, eligible_sessions=len(sessions), metric_groups=metric_groups,
        ))
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def evaluate_external_predictions(
    sessions, predictions: pd.DataFrame, *, cohort: str,
    persistence_min_consecutive: int = 2,
    metric_groups: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retains the original external prediction CSV contract and missing rows."""
    names = validate_metric_groups(metric_groups)
    missing = [column for column in PREDICTION_REQUIRED_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f'Missing prediction columns: {missing}')
    if predictions.duplicated(['example_id', 'model_name']).any():
        raise ValueError('Duplicate example_id/model_name prediction rows')
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    horizons = tuple(sorted(int(value) for value in predictions['horizon'].unique()))
    task_map = {
        task['example_id']: task
        for horizon in horizons for task in build_tasks(sessions, horizon)
    }
    provided = {}
    for record in predictions.to_dict('records'):
        task = task_map.get(record['example_id'])
        if task is None:
            raise ValueError(f"Unknown example_id: {record['example_id']}")
        if int(record['horizon']) != task['horizon']:
            raise ValueError(f"Horizon mismatch: {record['example_id']}")
        provided[(record['example_id'], str(record['model_name']))] = record
    rows = []
    requested_pairs = sorted({(str(r['model_name']), int(r['horizon']))
                              for r in predictions.to_dict('records')})
    for model_name, horizon in requested_pairs:
        for task in build_tasks(sessions, horizon):
            record = provided.get((task['example_id'], model_name))
            if record is None:
                rows.append(score_prediction(
                    task, None, cohort=cohort, model_name=model_name,
                    generation_status='missing',
                    persistence_min_consecutive=persistence_min_consecutive,
                    metric_groups=names,
                ))
            else:
                rows.append(score_prediction(
                    task, record.get('prediction'), cohort=cohort,
                    model_name=model_name, generation_status=str(record['generation_status']),
                    model_version=str(record.get('model_version', '')),
                    prompt_version=str(record.get('prompt_version', '')),
                    persistence_min_consecutive=persistence_min_consecutive,
                    metric_groups=names,
                ))
    detailed = pd.DataFrame(rows)
    summary = aggregate_prediction_scores(detailed, eligible_sessions=len(sessions),
                                          metric_groups=names)
    return detailed, summary


def evaluate_baselines(sessions: list[dict], horizons: tuple[int, ...], cohort: str) -> pd.DataFrame:
    return evaluate_predictions(sessions, horizons, cohort, BASELINE_PREDICTORS)


def run_experiment(config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run configured models and persist reusable predictions and scoring tables.

    Additional caching, run manifests and golden regression fixtures live under
    scaffolding/ and are intentionally NOT enabled in this minimal core.
    """
    config.validate()
    sessions = load_dataset(config.dataset)
    if not sessions:
        raise ValueError('No eligible sessions in selected dataset/cohort')
    predictors = load_predictors(config.models)
    records = generate_predictions(sessions, config.evaluation.horizons, predictors)
    if records.empty:
        raise ValueError('No tasks generated: check horizons versus trajectory length')
    detailed, summary = evaluate_external_predictions(
        sessions, records, cohort=config.dataset.cohort,
        persistence_min_consecutive=config.evaluation.persistence_min_consecutive,
        metric_groups=config.evaluation.metrics,
    )
    dest = Path(config.output.directory)
    dest.mkdir(parents=True, exist_ok=True)
    if config.output.save_predictions:
        records.to_csv(dest / 'predictions.csv', index=False)
    detailed.to_csv(dest / 'prediction_level.csv', index=False)
    summary.to_csv(dest / 'summary.csv', index=False)
    (dest / 'evaluation_protocol.json').write_text(
        json.dumps(protocol_specification(), indent=2), encoding='utf-8')
    # Save the effective config for inspection, not as a verified run manifest.
    (dest / 'effective_config.json').write_text(
        json.dumps(asdict(config), indent=2), encoding='utf-8')
    return records, detailed, summary
