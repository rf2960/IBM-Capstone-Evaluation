"""Extensible metric groups, preserving the v1 frozen scoring semantics.

Metric groups implement two simple callables: score(shared) -> per-task fields
and aggregate(group, scored) -> pooled fields. In particular, edit F1 and
stable F1 are recomputed from *pooled credit/counts*, NOT averaged task F1.
The base reference fields are computed independently of prediction validity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import numpy as np
import pandas as pd

from .edits import (
    extract_edits, edit_type, line_edit_cost, persistent_edits,
    edit_match_score, location_similarity, payload_similarity, operation_similarity,
    optimal_match_credit, prf_from_credit, whole_file_line_jaccard,
    durability_score,
)
from .tasks import is_exact_one_step_reversion


@dataclass(frozen=True)
class MetricGroup:
    score: Callable[[dict], dict]
    aggregate: Callable[[pd.DataFrame, pd.DataFrame], dict]
    description: str = ''


METRIC_REGISTRY: dict[str, MetricGroup] = {}
DEFAULT_METRICS = ('immediate', 'intervention', 'rework', 'persistence')


def register_metric_group(name: str, *, score: Callable[[dict], dict],
                          aggregate: Callable[[pd.DataFrame, pd.DataFrame], dict],
                          description: str = '') -> None:
    if not name or name in METRIC_REGISTRY:
        raise ValueError(f'Duplicate or invalid metric group: {name!r}')
    METRIC_REGISTRY[name] = MetricGroup(score, aggregate, description)


def validate_metric_groups(names: Iterable[str] | None) -> tuple[str, ...]:
    names = DEFAULT_METRICS if names is None else tuple(names)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate metric groups are not allowed')
    unknown = set(names) - set(METRIC_REGISTRY)
    if unknown:
        raise ValueError(f'Unknown metric groups: {sorted(unknown)}')
    return names


def _reference_fields(task: dict, persistence_min_consecutive: int) -> tuple[dict, dict]:
    if persistence_min_consecutive < 1:
        raise ValueError('persistence_min_consecutive must be >= 1')
    observed_edits = extract_edits(task['current'], task['target'])
    no_edit_target = not observed_edits
    reference_change_lines = line_edit_cost(task['current'], task['target'])
    future_observation_count = max(0, len(task['future']) - 1)
    reversion_eligible = task['horizon'] == 1 and bool(observed_edits) and future_observation_count > 0
    reversion_censored = task['horizon'] == 1 and bool(observed_edits) and future_observation_count == 0
    persistence_eligible = task['horizon'] == 1 and len(task['future']) >= persistence_min_consecutive
    persistence_censored = task['horizon'] == 1 and not persistence_eligible
    stable_observed_edits = (
        persistent_edits(observed_edits, task['future'], persistence_min_consecutive)
        if persistence_eligible else []
    )
    fields = {
        'future_observation_count': future_observation_count,
        'no_edit_target': no_edit_target,
        'observed_edit_type': edit_type(observed_edits),
        'observed_edit_blocks': len(observed_edits),
        'reference_change_lines': reference_change_lines,
        'rework_eligible': reference_change_lines > 0,
        'reversion_eligible': reversion_eligible,
        'exact_one_step_reversion': is_exact_one_step_reversion(task) if reversion_eligible else False,
        'reversion_censored': reversion_censored,
        'persistence_min_consecutive': persistence_min_consecutive,
        'persistence_eligible': persistence_eligible,
        'persistence_censored': persistence_censored,
        'persistence_observed_blocks': len(observed_edits) if persistence_eligible else 0,
        'persistence_censored_blocks': len(observed_edits) if persistence_censored else 0,
        'stable_reference_blocks': len(stable_observed_edits),
    }
    shared = {
        'task': task, 'observed_edits': observed_edits,
        'stable_observed_edits': stable_observed_edits, 'reference': fields,
    }
    return fields, shared


def score_prediction(
    task: dict,
    prediction: object,
    *,
    cohort: str,
    model_name: str,
    generation_status: str = 'ok',
    model_version: str = '',
    prompt_version: str = '',
    persistence_min_consecutive: int = 2,
    metric_groups: Iterable[str] | None = None,
) -> dict:
    """Score one prediction; unselected metric groups contribute no columns.

    Standard v1 default calculates all original metrics. Invalid generations
    retain base/reference counts but do not contribute predicted scores.
    """
    metric_groups = validate_metric_groups(metric_groups)
    ref_fields, shared = _reference_fields(task, persistence_min_consecutive)
    valid = generation_status == 'ok' and isinstance(prediction, str)
    base = {
        'cohort': cohort, 'model_name': model_name,
        'model_version': model_version, 'prompt_version': prompt_version,
        'example_id': task['example_id'], 'session_id': task['session_id'],
        'user_id': task.get('user_id', ''), 'source_model': task.get('source_model', ''),
        'outcome_timestamp': task.get('outcome_timestamp', ''),
        'anchor': task['anchor'], 'horizon': task['horizon'],
        'generation_status': generation_status, 'scored': bool(valid),
        **ref_fields,
    }
    shared['prediction'] = prediction if valid else None
    shared['valid'] = bool(valid)
    if valid:
        shared['predicted_edits'] = extract_edits(task['current'], prediction)
        shared['predicted_change_lines'] = line_edit_cost(task['current'], prediction)
        shared['remaining_rework_lines'] = line_edit_cost(prediction, task['target'])
    else:
        shared['predicted_edits'] = []
        shared['predicted_change_lines'] = float('nan')
        shared['remaining_rework_lines'] = float('nan')
    for name in metric_groups:
        fields = METRIC_REGISTRY[name].score(shared)
        overlap = set(fields) & set(base)
        if overlap:
            raise ValueError(f'Metric group {name} overwrites fields: {overlap}')
        base.update(fields)
    return base


def _score_immediate(c: dict) -> dict:
    if not c['valid']:
        return {
            'exact_match': float('nan'), 'whole_file_line_jaccard': float('nan'),
            'predicted_edit': False, 'predicted_edit_blocks': 0,
            'matched_edit_credit': 0.0, 'content_match_credit': 0.0,
            'location_match_credit': 0.0, 'operation_match_credit': 0.0,
            'edit_precision': float('nan'), 'edit_recall': float('nan'),
            'edit_f1': float('nan'), 'content_edit_f1': float('nan'),
            'location_edit_f1': float('nan'), 'operation_edit_f1': float('nan'),
        }
    pred, obs, task = c['predicted_edits'], c['observed_edits'], c['task']
    matched = optimal_match_credit(pred, obs, edit_match_score)
    content = optimal_match_credit(pred, obs, payload_similarity)
    location = optimal_match_credit(pred, obs, location_similarity)
    operation = optimal_match_credit(pred, obs, operation_similarity)
    precision, recall, f1 = prf_from_credit(matched, len(pred), len(obs))
    return {
        'exact_match': c['prediction'] == task['target'],
        'whole_file_line_jaccard': whole_file_line_jaccard(c['prediction'], task['target']),
        'predicted_edit': bool(pred), 'predicted_edit_blocks': len(pred),
        'matched_edit_credit': matched, 'content_match_credit': content,
        'location_match_credit': location, 'operation_match_credit': operation,
        'edit_precision': precision, 'edit_recall': recall, 'edit_f1': f1,
        'content_edit_f1': prf_from_credit(content, len(pred), len(obs))[2],
        'location_edit_f1': prf_from_credit(location, len(pred), len(obs))[2],
        'operation_edit_f1': prf_from_credit(operation, len(pred), len(obs))[2],
    }


def _score_intervention(c: dict) -> dict:
    return {'false_intervention': bool(c['predicted_edits']) and c['reference']['no_edit_target']}


def _score_rework(c: dict) -> dict:
    if not c['valid']:
        return {
            'predicted_change_lines': float('nan'), 'remaining_rework_lines': float('nan'),
            'rework_reduction': float('nan'), 'rework_improved': False,
            'rework_worsened': False, 'unnecessary_change_lines': float('nan'),
        }
    reference_cost = c['reference']['reference_change_lines']
    predicted_cost = c['predicted_change_lines']
    remaining_cost = c['remaining_rework_lines']
    return {
        'predicted_change_lines': predicted_cost,
        'remaining_rework_lines': remaining_cost,
        'rework_reduction': ((reference_cost - remaining_cost) / reference_cost
                             if reference_cost else float('nan')),
        'rework_improved': reference_cost > 0 and remaining_cost < reference_cost,
        'rework_worsened': reference_cost > 0 and remaining_cost > reference_cost,
        'unnecessary_change_lines': predicted_cost if c['reference']['no_edit_target'] else 0,
    }


def _score_persistence(c: dict) -> dict:
    if not c['valid']:
        return {
            'stable_predicted_blocks': 0, 'stable_match_credit': 0.0,
            'stable_precision': float('nan'), 'stable_recall': float('nan'),
            'stable_f1': float('nan'), 'durability': float('nan'),
        }
    eligible = c['reference']['persistence_eligible']
    pred = c['predicted_edits']
    stable_count = len(pred) if eligible else 0
    credit = (optimal_match_credit(pred, c['stable_observed_edits'], edit_match_score)
              if eligible else 0.0)
    p, r, f = prf_from_credit(credit, stable_count, len(c['stable_observed_edits']))
    task = c['task']
    return {
        'stable_predicted_blocks': stable_count, 'stable_match_credit': credit,
        'stable_precision': p, 'stable_recall': r, 'stable_f1': f,
        'durability': durability_score(task['current'], c['prediction'], task['future']),
    }


def _aggregate_base(group: pd.DataFrame, scored: pd.DataFrame, eligible_sessions: int,
                    cohort: str, horizon: int, model_name: str) -> dict:
    n_tasks, n_scored = len(group), len(scored)
    reversion_eligible = group[group['reversion_eligible']]
    reversion_count = int(reversion_eligible['exact_one_step_reversion'].sum())
    return {
        'cohort': cohort, 'horizon': int(horizon), 'baseline': model_name,
        'eligible_sessions': eligible_sessions, 'n_tasks': n_tasks,
        'scored_predictions': n_scored, 'invalid_predictions': n_tasks - n_scored,
        'prediction_coverage': n_scored / n_tasks if n_tasks else float('nan'),
        'no_edit_target_rate': float(group['no_edit_target'].mean()) if n_tasks else float('nan'),
        'reversion_eligible_tasks': len(reversion_eligible),
        'reversion_count': reversion_count,
        'reversion_rate': reversion_count / len(reversion_eligible) if len(reversion_eligible) else float('nan'),
        'reversion_censored_tasks': int(group['reversion_censored'].sum()),
    }


def _aggregate_immediate(group: pd.DataFrame, scored: pd.DataFrame) -> dict:
    n_scored = len(scored)
    pred_count = int(scored['predicted_edit_blocks'].sum())
    obs_count = int(scored['observed_edit_blocks'].sum())
    p, r, f = prf_from_credit(float(scored['matched_edit_credit'].sum()), pred_count, obs_count)
    def pooled(key: str) -> float:
        return prf_from_credit(float(scored[key].sum()), pred_count, obs_count)[2]
    return {
        'exact_count': int(scored['exact_match'].sum()),
        'exact_match': float(scored['exact_match'].mean()) if n_scored else float('nan'),
        'whole_file_line_jaccard': float(scored['whole_file_line_jaccard'].mean()) if n_scored else float('nan'),
        'immediate_edit_precision': p, 'immediate_edit_recall': r,
        'location_content_edit_f1': f,
        'content_edit_f1': pooled('content_match_credit'),
        'location_edit_f1': pooled('location_match_credit'),
        'operation_edit_f1': pooled('operation_match_credit'),
        'predicted_edit_blocks': pred_count, 'observed_edit_blocks': obs_count,
        'predicted_edit_rate': float(scored['predicted_edit'].mean()) if n_scored else float('nan'),
    }


def _aggregate_intervention(group: pd.DataFrame, scored: pd.DataFrame) -> dict:
    no_edit_scored = scored[scored['no_edit_target']]
    return {
        'false_intervention_eligible_tasks': len(no_edit_scored),
        'false_intervention_count': int(no_edit_scored['false_intervention'].sum()),
        'false_intervention_rate': (float(no_edit_scored['false_intervention'].mean())
                                    if len(no_edit_scored) else float('nan')),
    }


def _aggregate_rework(group: pd.DataFrame, scored: pd.DataFrame) -> dict:
    eligible = scored[scored['rework_eligible']]
    no_edit_scored = scored[scored['no_edit_target']]
    ref = float(eligible['reference_change_lines'].sum())
    remain = float(eligible['remaining_rework_lines'].sum())
    def avg(column: str) -> float:
        return float(eligible[column].mean()) if len(eligible) else float('nan')
    return {
        'rework_eligible_tasks': len(eligible),
        'reference_change_lines_total': int(ref),
        'remaining_rework_lines_total': int(remain),
        'pooled_rework_reduction': (ref - remain) / ref if ref else float('nan'),
        'mean_reference_change_lines': avg('reference_change_lines'),
        'mean_predicted_change_lines': avg('predicted_change_lines'),
        'mean_remaining_rework_lines': avg('remaining_rework_lines'),
        'mean_rework_reduction': avg('rework_reduction'),
        'rework_improved_rate': avg('rework_improved'),
        'rework_worsened_rate': avg('rework_worsened'),
        'mean_unnecessary_change_lines': (float(no_edit_scored['unnecessary_change_lines'].mean())
                                           if len(no_edit_scored) else float('nan')),
    }


def _aggregate_persistence(group: pd.DataFrame, scored: pd.DataFrame) -> dict:
    eligible = scored[scored['persistence_eligible']]
    censored = scored[scored['persistence_censored']]
    observed = int(eligible['persistence_observed_blocks'].sum())
    censored_blocks = int(censored['persistence_censored_blocks'].sum())
    stable_ref = int(eligible['stable_reference_blocks'].sum())
    stable_pred = int(eligible['stable_predicted_blocks'].sum())
    credit = float(eligible['stable_match_credit'].sum())
    p, r, f = prf_from_credit(credit, stable_pred, stable_ref)
    return {
        'durability_n': int(scored['durability'].notna().sum()),
        'mean_durability': (float(scored['durability'].mean()) if scored['durability'].notna().any() else float('nan')),
        'persistence_min_consecutive': int(group['persistence_min_consecutive'].iloc[0]),
        'persistence_eligible_tasks': len(eligible),
        'persistence_censored_tasks': len(censored),
        'persistence_observed_blocks': observed,
        'persistence_censored_blocks': censored_blocks,
        'stable_reference_blocks': stable_ref,
        'stable_reference_rate': stable_ref / observed if observed else float('nan'),
        'stable_predicted_blocks': stable_pred,
        'stable_precision': p, 'stable_recall': r, 'stable_f1': f,
    }


register_metric_group('immediate', score=_score_immediate, aggregate=_aggregate_immediate,
                      description='Immediate file/edit match and block precision/recall')
register_metric_group('intervention', score=_score_intervention, aggregate=_aggregate_intervention,
                      description='False intervention on no-edit targets')
register_metric_group('rework', score=_score_rework, aggregate=_aggregate_rework,
                      description='Residual line-edit cost and work reduction')
register_metric_group('persistence', score=_score_persistence, aggregate=_aggregate_persistence,
                      description='Frozen consecutive-state persistence and durability')


def aggregate_prediction_scores(detailed: pd.DataFrame, *, eligible_sessions: int,
                                metric_groups: Iterable[str] | None = None) -> pd.DataFrame:
    names = validate_metric_groups(metric_groups)
    if detailed.empty:
        return pd.DataFrame()
    rows = []
    for (cohort, horizon, model_name), group in detailed.groupby(
        ['cohort', 'horizon', 'model_name'], sort=False
    ):
        scored = group[group['scored']]
        result = _aggregate_base(group, scored, eligible_sessions, cohort, horizon, model_name)
        for name in names:
            addition = METRIC_REGISTRY[name].aggregate(group, scored)
            overlap = set(result) & set(addition)
            if overlap:
                raise ValueError(f'Metric group {name} overwrites aggregate: {overlap}')
            result.update(addition)
        rows.append(result)
    return pd.DataFrame(rows)
