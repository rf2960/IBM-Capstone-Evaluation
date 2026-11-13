from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from statistics import mean
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


METRIC_DEFINITIONS = {
    "cohort": "The preprocessing policy used to select trajectory groups.",
    "baseline": "The predictor being evaluated: no-change or repeat-last-edit.",
    "horizon": "How many recorded states ahead of the current state the target is.",
    "eligible_sessions": "Number of trajectory groups that contribute to this evaluation run.",
    "n_tasks": "Number of sliding prediction windows; this is the denominator for task-level rates.",
    "scored_predictions": "Predictions with status ok and a text output.",
    "invalid_predictions": "Requested predictions that could not be scored.",
    "prediction_coverage": "scored_predictions / n_tasks.",
    "exact_count": "Scored predictions whose full code string exactly equals the target.",
    "exact_match": "exact_count / scored_predictions.",
    "whole_file_line_jaccard": "Mean set overlap between non-empty stripped lines in the predicted and target full files.",
    "no_edit_target_rate": "Fraction of tasks where current and target contain no line-diff edit block.",
    "false_intervention_eligible_tasks": "Scored tasks whose target contains no observed edit.",
    "false_intervention_count": "No-edit target tasks where the predictor proposed at least one edit block.",
    "false_intervention_rate": "false_intervention_count / false_intervention_eligible_tasks.",
    "reversion_eligible_tasks": "Reference-trajectory horizon-1 edited tasks that also have one recorded state after the target.",
    "reversion_count": "Reference-trajectory tasks whose state after the target exactly returns to the pre-edit current code.",
    "reversion_rate": "Observed reference reversion_count / reversion_eligible_tasks; reported only at horizon 1 and not a model score.",
    "reversion_censored_tasks": "Edited reference-trajectory horizon-1 tasks with no state after the target.",
    "predicted_edit_blocks": "Total insert, delete, or replace blocks extracted from current to prediction.",
    "observed_edit_blocks": "Total insert, delete, or replace blocks extracted from current to target.",
    "predicted_edit_rate": "Fraction of tasks where current to prediction contains at least one edit block; activity, not accuracy.",
    "immediate_edit_precision": "Total location-and-content match credit divided by predicted edit blocks.",
    "immediate_edit_recall": "Total location-and-content match credit divided by observed edit blocks.",
    "location_content_edit_f1": "Harmonic mean of immediate edit precision and recall.",
    "content_edit_f1": "Edit-block F1 using payload similarity without requiring the correct location.",
    "location_edit_f1": "Edit-block F1 using source-line location without requiring matching content.",
    "operation_edit_f1": "Edit-block F1 requiring the same insert, delete, or replace operation.",
    "reference_change_lines": "Added plus deleted lines needed to move from current code to the observed target; replacements count both sides.",
    "predicted_change_lines": "Added plus deleted lines introduced by the prediction relative to current code.",
    "remaining_rework_lines": "Added plus deleted lines still needed to move from the prediction to the observed target.",
    "rework_reduction": "(reference_change_lines - remaining_rework_lines) / reference_change_lines; defined only when the observed target changes.",
    "pooled_rework_reduction": "(total reference_change_lines - total remaining_rework_lines) / total reference_change_lines over changed-target tasks.",
    "rework_improved_rate": "Fraction of changed-target tasks where prediction-to-target line cost is lower than current-to-target cost.",
    "rework_worsened_rate": "Fraction of changed-target tasks where prediction-to-target line cost is higher than current-to-target cost.",
    "mean_unnecessary_change_lines": "Mean predicted line-change cost on observed no-edit targets.",
    "durability_n": "Tasks with a non-empty predicted edit and therefore a defined durability score.",
    "mean_durability": "Mean weighted retention of predicted edit payloads near their original location over up to five future states.",
    "persistence_min_consecutive": "Required consecutive observed states, including the target, used to call an edit persistent.",
    "persistence_eligible_tasks": "Horizon-1 tasks with enough observed states to apply the persistence rule.",
    "persistence_censored_tasks": "Horizon-1 tasks without enough follow-up to apply the persistence rule.",
    "persistence_observed_blocks": "Observed target edit blocks on persistence-eligible tasks.",
    "persistence_censored_blocks": "Observed target edit blocks excluded because follow-up is too short.",
    "stable_reference_blocks": "Observed edit blocks retained for the required consecutive states.",
    "stable_reference_rate": "stable_reference_blocks / persistence_observed_blocks.",
    "stable_predicted_blocks": "Predicted edit blocks on persistence-eligible tasks.",
    "stable_precision": "Location-and-content match credit to stable reference blocks / stable predicted blocks.",
    "stable_recall": "Location-and-content match credit to stable reference blocks / stable reference blocks.",
    "stable_f1": "F1 combining stable precision and stable recall.",
}

PREDICTION_REQUIRED_COLUMNS = (
    "example_id",
    "model_name",
    "horizon",
    "prediction",
    "generation_status",
)

PROTOCOL_VERSION = "1.0-phase1-freeze"
PRIMARY_SCORECARD_COLUMNS = (
    "baseline",
    "n_tasks",
    "prediction_coverage",
    "exact_match",
    "location_content_edit_f1",
    "false_intervention_rate",
    "pooled_rework_reduction",
    "stable_f1",
)

ERROR_TAXONOMY = {
    "correct_abstention": "Observed next state contains no substantive edit; abstaining is correct.",
    "partial_edit": "Prediction overlaps the target edit but does not recover the complete change.",
    "wrong_content": "Prediction changes the wrong text or introduces unsupported content.",
    "wrong_location": "Predicted content is plausible but placed at the wrong source location.",
    "wrong_operation": "Insert, delete, or replace action does not match the observed operation.",
    "wrong_timing": "Edit is plausible in the trajectory but not at the evaluated next state.",
    "over_editing": "Prediction changes code beyond what the observed target requires.",
    "uncertain_reference": "Observed target is transient or otherwise weak evidence of durable intent.",
    "formatting_only_exclude": "Transition contains only formatting or whitespace evidence and is excluded from semantic review.",
}

CASE_AUDIT_DECISIONS = {
    "C01": ("correct_abstention", "javascript", "standalone_parse_pass"),
    "C02": ("correct_abstention", "unknown", "language_unknown"),
    "C03": ("correct_abstention", "unknown", "language_unknown"),
    "C04": ("correct_abstention", "unknown", "language_unknown"),
    "C05": ("partial_edit", "javascript", "standalone_parse_pass"),
    "C06": ("formatting_only_exclude", "unknown", "language_unknown"),
    "C07": ("formatting_only_exclude", "unknown", "language_unknown"),
    "C08": ("over_editing", "python", "fragment_or_parse_fail"),
    "C09": ("uncertain_reference", "unknown", "language_unknown"),
    "C10": ("uncertain_reference", "python", "fragment_or_parse_fail"),
    "C11": ("uncertain_reference", "unknown", "language_unknown"),
    "C12": ("uncertain_reference", "unknown", "language_unknown"),
    "C13": ("wrong_content", "unknown", "language_unknown"),
    "C14": ("uncertain_reference", "python", "fragment_or_parse_fail"),
    "C15": ("partial_edit", "unknown", "language_unknown"),
    "C16": ("wrong_content", "javascript", "standalone_parse_pass"),
    "C17": ("wrong_operation", "unknown", "language_unknown"),
    "C18": ("over_editing", "python", "fragment_or_parse_fail"),
    "C19": ("over_editing", "unknown", "language_unknown"),
    "C20": ("over_editing", "python", "fragment_or_parse_fail"),
}


@dataclass(frozen=True)
class Edit:
    tag: str
    start: int
    end: int
    payload: tuple[str, ...]

    @property
    def anchor(self) -> int:
        return self.start


def clean_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()]


def line_set(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if line.strip()}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def whole_file_line_jaccard(left: str, right: str) -> float:
    return jaccard(line_set(left), line_set(right))


def line_edit_cost(source: str, destination: str) -> int:
    """Count added plus deleted lines in a shortest line-level diff.

    A replacement counts both the removed source lines and inserted destination
    lines. This makes oversized rewrites visibly more expensive than a compact
    patch while remaining language agnostic.
    """
    source_lines = clean_lines(source)
    destination_lines = clean_lines(destination)
    matcher = SequenceMatcher(None, source_lines, destination_lines, autojunk=False)
    cost = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            cost += (i2 - i1) + (j2 - j1)
        elif tag == "delete":
            cost += i2 - i1
        elif tag == "insert":
            cost += j2 - j1
    return cost


def extract_edits(source: str, destination: str) -> list[Edit]:
    source_lines = clean_lines(source)
    destination_lines = clean_lines(destination)
    matcher = SequenceMatcher(None, source_lines, destination_lines, autojunk=False)
    edits: list[Edit] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        payload = (
            tuple(destination_lines[j1:j2])
            if tag in {"insert", "replace"}
            else tuple(source_lines[i1:i2])
        )
        edits.append(Edit(tag=tag, start=i1, end=i2, payload=payload))
    return edits


def location_similarity(left: Edit, right: Edit, tolerant: bool = False) -> float:
    distance = abs(left.anchor - right.anchor)
    if distance == 0:
        return 1.0
    if tolerant and distance == 1:
        return 0.5
    return 0.0


def payload_similarity(left: Edit, right: Edit) -> float:
    left_text = "\n".join(line.strip() for line in left.payload if line.strip())
    right_text = "\n".join(line.strip() for line in right.payload if line.strip())
    if left.tag == right.tag == "delete" and not left_text and not right_text:
        return 1.0
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()


def edit_match_score(left: Edit, right: Edit, tolerant: bool = False) -> float:
    return location_similarity(left, right, tolerant=tolerant) * payload_similarity(left, right)


def operation_similarity(left: Edit, right: Edit) -> float:
    return float(left.tag == right.tag)


def greedy_match_credit(
    predicted: list[Edit],
    observed: list[Edit],
    scorer: Callable[[Edit, Edit], float],
) -> float:
    """One-to-one maximum-first credit under a supplied edit comparison."""
    candidates = sorted(
        (
            (scorer(predicted[p], observed[o]), p, o)
            for p in range(len(predicted))
            for o in range(len(observed))
        ),
        reverse=True,
    )
    used_predicted: set[int] = set()
    used_observed: set[int] = set()
    credit = 0.0
    for score, p_index, o_index in candidates:
        if score <= 0:
            break
        if p_index in used_predicted or o_index in used_observed:
            continue
        used_predicted.add(p_index)
        used_observed.add(o_index)
        credit += score
    return credit


def optimal_match_credit(
    predicted: list[Edit],
    observed: list[Edit],
    scorer: Callable[[Edit, Edit], float],
) -> float:
    """Maximum total one-to-one edit credit via linear assignment."""
    if not predicted or not observed:
        return 0.0
    scores = np.array(
        [
            [scorer(predicted_edit, observed_edit) for observed_edit in observed]
            for predicted_edit in predicted
        ],
        dtype=float,
    )
    predicted_indices, observed_indices = linear_sum_assignment(-scores)
    return float(scores[predicted_indices, observed_indices].sum())


def prf_from_credit(
    credit: float,
    predicted_count: int,
    observed_count: int,
) -> tuple[float, float, float]:
    """Return precision, recall, and F1 with an explicit zero-division policy."""
    precision = credit / predicted_count if predicted_count else float("nan")
    recall = credit / observed_count if observed_count else float("nan")
    denominator = predicted_count + observed_count
    f1 = 2 * credit / denominator if denominator else float("nan")
    return precision, recall, f1


def soft_edit_prf(
    predicted: list[Edit],
    observed: list[Edit],
    *,
    tolerant: bool = False,
) -> tuple[float, float, float]:
    if not predicted and not observed:
        return 1.0, 1.0, 1.0
    if not predicted or not observed:
        return 0.0, 0.0, 0.0

    credit = optimal_match_credit(
        predicted,
        observed,
        lambda left, right: edit_match_score(
            left, right, tolerant=tolerant
        ),
    )
    return prf_from_credit(credit, len(predicted), len(observed))


def edit_type(edits: list[Edit]) -> str:
    if not edits:
        return "no_edit"
    tags = {edit.tag for edit in edits}
    return next(iter(tags)) if len(tags) == 1 else "mixed"


def persistent_edits(
    edits: list[Edit],
    future_states: list[str],
    min_consecutive: int,
) -> list[Edit]:
    """Edits retained in the target and required following observed states."""
    if min_consecutive < 1:
        raise ValueError("min_consecutive must be at least 1")
    if len(future_states) < min_consecutive:
        return []
    window = future_states[:min_consecutive]
    return [
        edit
        for edit in edits
        if all(
            predicted_edit_retention([edit], state) >= 1.0 - 1e-12
            for state in window
        )
    ]


def predicted_edit_retention(predicted: list[Edit], future_state: str) -> float:
    """Check whether predicted payload survives near its original edit location.

    This deliberately does not penalize additional unrelated future edits.
    A two-line location window allows small shifts caused by nearby insertions.
    """
    if not predicted:
        return 1.0
    future_lines = clean_lines(future_state)
    edit_scores = []
    for edit in predicted:
        payload = [line.strip() for line in edit.payload if line.strip()]
        start = max(0, edit.anchor - 2)
        stop = min(len(future_lines), edit.anchor + max(1, len(payload)) + 3)
        window = [line.strip() for line in future_lines[start:stop] if line.strip()]
        if not payload:
            edit_scores.append(1.0)
            continue
        similarities = [float(line in window) for line in payload]
        payload_score = mean(similarities)
        edit_scores.append(1.0 - payload_score if edit.tag == "delete" else payload_score)
    return mean(edit_scores)


def durability_score(
    current: str,
    prediction: str,
    future_states: list[str],
    decay: float = 0.8,
) -> float:
    predicted_edits = extract_edits(current, prediction)
    if not predicted_edits:
        return float("nan")
    future_states = future_states[:5]
    weights = [decay**index for index in range(len(future_states))]
    scores = [predicted_edit_retention(predicted_edits, future) for future in future_states]
    return sum(weight * score for weight, score in zip(weights, scores)) / sum(weights)


def middle_delta(previous: str, current: str) -> tuple[int, int, int, str, str]:
    """Return the one contiguous character span changed from previous to current."""
    prefix = 0
    while prefix < min(len(previous), len(current)) and previous[prefix] == current[prefix]:
        prefix += 1
    suffix = 0
    max_suffix = min(len(previous) - prefix, len(current) - prefix)
    while suffix < max_suffix and previous[-1 - suffix] == current[-1 - suffix]:
        suffix += 1
    previous_end = len(previous) - suffix if suffix else len(previous)
    current_end = len(current) - suffix if suffix else len(current)
    old_middle = previous[prefix:previous_end]
    new_middle = current[prefix:current_end]
    return prefix, previous_end, current_end, old_middle, new_middle


def repeat_last_edit(previous: str, current: str) -> str:
    prefix, _, current_end, old_middle, new_middle = middle_delta(previous, current)
    if not old_middle and new_middle:
        return current[:current_end] + new_middle + current[current_end:]
    replacement_end = min(prefix + len(old_middle), len(current))
    return current[:prefix] + new_middle + current[replacement_end:]


def repeat_last_edit_category(previous: str, current: str) -> str:
    """Explain whether and why the repeat heuristic changes the current code."""
    if previous == current:
        return "no_prior_change"
    prediction = repeat_last_edit(previous, current)
    if extract_edits(current, prediction):
        return "proposes_line_edit"
    if prediction != current:
        return "whitespace_only_change"
    _, _, _, old_middle, new_middle = middle_delta(previous, current)
    if old_middle and new_middle and len(old_middle) == len(new_middle):
        return "idempotent_replacement"
    if old_middle and not new_middle:
        return "deletion_replay_noop"
    return "other_noop"


def load_sessions(frame: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    strict_sessions: list[dict] = []
    drop_none_sessions: list[dict] = []
    state_type_counts: dict[str, int] = {}
    rows_with_non_string = 0
    rows_with_short_state_list = 0
    total_non_string_elements = 0
    none_positions = {"leading": 0, "interior": 0, "trailing": 0}

    for row in frame.sort_values("outcome_completion_id").itertuples(index=False):
        states = list(row.states)
        non_string_types = [type(state).__name__ for state in states if not isinstance(state, str)]
        for name in non_string_types:
            state_type_counts[name] = state_type_counts.get(name, 0) + 1
        if non_string_types:
            rows_with_non_string += 1
            total_non_string_elements += len(non_string_types)
            none_positions["leading"] += int(states[0] is None)
            none_positions["trailing"] += int(states[-1] is None)
            none_positions["interior"] += int(any(state is None for state in states[1:-1]))
        if len(states) < 3:
            rows_with_short_state_list += 1
        if len(states) >= 3 and not non_string_types:
            strict_sessions.append(
                {
                    "session_id": row.outcome_completion_id,
                    "user_id": row.user_id,
                    "source_model": row.model,
                    "outcome_timestamp": row.outcome_timestamp,
                    "states": states,
                }
            )
        string_states = [state for state in states if isinstance(state, str)]
        if len(string_states) >= 3:
            drop_none_sessions.append(
                {
                    "session_id": row.outcome_completion_id,
                    "user_id": row.user_id,
                    "source_model": row.model,
                    "outcome_timestamp": row.outcome_timestamp,
                    "states": string_states,
                }
            )

    audit = {
        "raw_trajectory_groups": int(len(frame)),
        "total_snapshots": int(frame["n_snapshots"].sum()),
        "total_states_including_s0": int((frame["n_snapshots"] + 1).sum()),
        "unique_users": int(frame["user_id"].nunique()),
        "unique_source_models": int(frame["model"].nunique()),
        "snapshot_count_mean": float(frame["n_snapshots"].mean()),
        "snapshot_count_median": float(frame["n_snapshots"].median()),
        "snapshot_count_max": int(frame["n_snapshots"].max()),
        "states_equals_s0_plus_snapshots_for_all_rows": bool(
            all(
                list(row.states) == [row.s0] + list(row.snapshots)
                for row in frame.itertuples(index=False)
            )
        ),
        "strict_eligible_sessions": len(strict_sessions),
        "eligible_after_dropping_none": len(drop_none_sessions),
        "additional_sessions_if_none_is_dropped": len(drop_none_sessions) - len(strict_sessions),
        "rows_with_fewer_than_3_states": rows_with_short_state_list,
        "rows_with_non_string_state": rows_with_non_string,
        "total_non_string_state_elements": total_non_string_elements,
        "non_string_element_types": state_type_counts,
        "rows_with_none_by_position": none_positions,
        "eligibility_rule": "at least 3 states and every state is a string",
    }
    return strict_sessions, drop_none_sessions, audit


def load_segmented_sessions(
    frame: pd.DataFrame,
    *,
    id_column: str = "segment_id",
    states_column: str = "states",
) -> list[dict]:
    """Load a future normalized/segmented table without changing metric code."""
    sessions = []
    for row in frame.to_dict("records"):
        states = list(row[states_column])
        if len(states) >= 3 and all(isinstance(state, str) for state in states):
            sessions.append({"session_id": row[id_column], "states": states})
    return sessions


def build_tasks(sessions: list[dict], horizon: int):
    for session in sessions:
        states = session["states"]
        for anchor in range(1, len(states) - horizon):
            example_id = f"{session['session_id']}::a{anchor}::h{horizon}"
            yield {
                "example_id": example_id,
                "session_id": session["session_id"],
                "user_id": session.get("user_id", ""),
                "source_model": session.get("source_model", ""),
                "outcome_timestamp": session.get("outcome_timestamp", ""),
                "anchor": anchor,
                "horizon": horizon,
                "previous": states[anchor - 1],
                "current": states[anchor],
                "target": states[anchor + horizon],
                "future": states[anchor + horizon :],
            }


def build_model_inputs(sessions: list[dict], horizons: tuple[int, ...]) -> pd.DataFrame:
    """Export only information available at prediction time."""
    rows = []
    for horizon in horizons:
        for task in build_tasks(sessions, horizon):
            rows.append(
                {
                    "example_id": task["example_id"],
                    "session_id": task["session_id"],
                    "anchor": task["anchor"],
                    "horizon": horizon,
                    "previous": task["previous"],
                    "current": task["current"],
                }
            )
    return pd.DataFrame(rows)


def repeat_activity_audit(sessions: list[dict], horizon: int = 1) -> pd.DataFrame:
    """Count why repeat-last-edit does or does not propose a new edit."""
    tasks = list(build_tasks(sessions, horizon))
    counts: dict[str, int] = {}
    for task in tasks:
        category = repeat_last_edit_category(task["previous"], task["current"])
        counts[category] = counts.get(category, 0) + 1
    order = [
        "proposes_line_edit",
        "no_prior_change",
        "idempotent_replacement",
        "deletion_replay_noop",
        "whitespace_only_change",
        "other_noop",
    ]
    return pd.DataFrame(
        [
            {
                "horizon": horizon,
                "category": category,
                "count": counts.get(category, 0),
                "rate": counts.get(category, 0) / len(tasks) if tasks else float("nan"),
            }
            for category in order
            if counts.get(category, 0)
        ]
    )


def is_exact_one_step_reversion(task: dict) -> bool:
    """True when the state after an edited h=1 target returns exactly to current."""
    return (
        task["target"] != task["current"]
        and len(task["future"]) >= 2
        and task["future"][1] == task["current"]
    )


PredictionFn = Callable[[dict], str]


def predict_no_change(task: dict) -> str:
    return task["current"]


def predict_repeat_last_edit(task: dict) -> str:
    return repeat_last_edit(task["previous"], task["current"])


BASELINE_PREDICTORS = {
    "no_change": predict_no_change,
    "repeat_last_edit": predict_repeat_last_edit,
}


def score_prediction(
    task: dict,
    prediction: object,
    *,
    cohort: str,
    model_name: str,
    generation_status: str = "ok",
    model_version: str = "",
    prompt_version: str = "",
    persistence_min_consecutive: int = 2,
) -> dict:
    """Score one prediction and retain every denominator used later."""
    observed_edits = extract_edits(task["current"], task["target"])
    no_edit_target = not observed_edits
    reference_change_lines = line_edit_cost(task["current"], task["target"])
    future_observation_count = max(0, len(task["future"]) - 1)
    reversion_eligible = (
        task["horizon"] == 1
        and bool(observed_edits)
        and future_observation_count > 0
    )
    reversion_censored = (
        task["horizon"] == 1
        and bool(observed_edits)
        and future_observation_count == 0
    )
    persistence_eligible = (
        task["horizon"] == 1
        and len(task["future"]) >= persistence_min_consecutive
    )
    persistence_censored = (
        task["horizon"] == 1 and not persistence_eligible
    )
    stable_observed_edits = (
        persistent_edits(
            observed_edits,
            task["future"],
            persistence_min_consecutive,
        )
        if persistence_eligible
        else []
    )
    base = {
        "cohort": cohort,
        "model_name": model_name,
        "model_version": model_version,
        "prompt_version": prompt_version,
        "example_id": task["example_id"],
        "session_id": task["session_id"],
        "user_id": task.get("user_id", ""),
        "source_model": task.get("source_model", ""),
        "outcome_timestamp": task.get("outcome_timestamp", ""),
        "anchor": task["anchor"],
        "horizon": task["horizon"],
        "generation_status": generation_status,
        "scored": False,
        "future_observation_count": future_observation_count,
        "no_edit_target": no_edit_target,
        "observed_edit_type": edit_type(observed_edits),
        "observed_edit_blocks": len(observed_edits),
        "reference_change_lines": reference_change_lines,
        "rework_eligible": reference_change_lines > 0,
        "reversion_eligible": reversion_eligible,
        "exact_one_step_reversion": (
            is_exact_one_step_reversion(task) if reversion_eligible else False
        ),
        "reversion_censored": reversion_censored,
        "persistence_min_consecutive": persistence_min_consecutive,
        "persistence_eligible": persistence_eligible,
        "persistence_censored": persistence_censored,
        "persistence_observed_blocks": (
            len(observed_edits) if persistence_eligible else 0
        ),
        "persistence_censored_blocks": (
            len(observed_edits) if persistence_censored else 0
        ),
        "stable_reference_blocks": len(stable_observed_edits),
    }
    if generation_status != "ok" or not isinstance(prediction, str):
        return {
            **base,
            "exact_match": float("nan"),
            "whole_file_line_jaccard": float("nan"),
            "predicted_edit": False,
            "predicted_edit_blocks": 0,
            "matched_edit_credit": 0.0,
            "content_match_credit": 0.0,
            "location_match_credit": 0.0,
            "operation_match_credit": 0.0,
            "edit_precision": float("nan"),
            "edit_recall": float("nan"),
            "edit_f1": float("nan"),
            "content_edit_f1": float("nan"),
            "location_edit_f1": float("nan"),
            "operation_edit_f1": float("nan"),
            "false_intervention": False,
            "predicted_change_lines": float("nan"),
            "remaining_rework_lines": float("nan"),
            "rework_reduction": float("nan"),
            "rework_improved": False,
            "rework_worsened": False,
            "unnecessary_change_lines": float("nan"),
            "stable_predicted_blocks": 0,
            "stable_match_credit": 0.0,
            "stable_precision": float("nan"),
            "stable_recall": float("nan"),
            "stable_f1": float("nan"),
            "durability": float("nan"),
        }

    predicted_edits = extract_edits(task["current"], prediction)
    predicted_change_lines = line_edit_cost(task["current"], prediction)
    remaining_rework_lines = line_edit_cost(prediction, task["target"])
    rework_reduction = (
        (reference_change_lines - remaining_rework_lines)
        / reference_change_lines
        if reference_change_lines
        else float("nan")
    )
    matched_credit = optimal_match_credit(
        predicted_edits, observed_edits, edit_match_score
    )
    content_match_credit = optimal_match_credit(
        predicted_edits, observed_edits, payload_similarity
    )
    location_match_credit = optimal_match_credit(
        predicted_edits, observed_edits, location_similarity
    )
    operation_match_credit = optimal_match_credit(
        predicted_edits, observed_edits, operation_similarity
    )
    edit_precision, edit_recall, edit_f1 = prf_from_credit(
        matched_credit, len(predicted_edits), len(observed_edits)
    )
    content_edit_f1 = prf_from_credit(
        content_match_credit, len(predicted_edits), len(observed_edits)
    )[2]
    location_edit_f1 = prf_from_credit(
        location_match_credit, len(predicted_edits), len(observed_edits)
    )[2]
    operation_edit_f1 = prf_from_credit(
        operation_match_credit, len(predicted_edits), len(observed_edits)
    )[2]
    stable_predicted_blocks = (
        len(predicted_edits) if persistence_eligible else 0
    )
    stable_match_credit = (
        optimal_match_credit(
            predicted_edits, stable_observed_edits, edit_match_score
        )
        if persistence_eligible
        else 0.0
    )
    stable_precision, stable_recall, stable_f1 = prf_from_credit(
        stable_match_credit,
        stable_predicted_blocks,
        len(stable_observed_edits),
    )
    return {
        **base,
        "scored": True,
        "exact_match": prediction == task["target"],
        "whole_file_line_jaccard": whole_file_line_jaccard(
            prediction, task["target"]
        ),
        "predicted_edit": bool(predicted_edits),
        "predicted_edit_blocks": len(predicted_edits),
        "matched_edit_credit": matched_credit,
        "content_match_credit": content_match_credit,
        "location_match_credit": location_match_credit,
        "operation_match_credit": operation_match_credit,
        "edit_precision": edit_precision,
        "edit_recall": edit_recall,
        "edit_f1": edit_f1,
        "content_edit_f1": content_edit_f1,
        "location_edit_f1": location_edit_f1,
        "operation_edit_f1": operation_edit_f1,
        "false_intervention": bool(predicted_edits) and no_edit_target,
        "predicted_change_lines": predicted_change_lines,
        "remaining_rework_lines": remaining_rework_lines,
        "rework_reduction": rework_reduction,
        "rework_improved": (
            reference_change_lines > 0
            and remaining_rework_lines < reference_change_lines
        ),
        "rework_worsened": (
            reference_change_lines > 0
            and remaining_rework_lines > reference_change_lines
        ),
        "unnecessary_change_lines": (
            predicted_change_lines if no_edit_target else 0
        ),
        "stable_predicted_blocks": stable_predicted_blocks,
        "stable_match_credit": stable_match_credit,
        "stable_precision": stable_precision,
        "stable_recall": stable_recall,
        "stable_f1": stable_f1,
        "durability": durability_score(
            task["current"], prediction, task["future"]
        ),
    }


def evaluate_predictors_detailed(
    sessions: list[dict],
    horizons: tuple[int, ...],
    cohort: str,
    predictors: dict[str, PredictionFn],
    *,
    persistence_min_consecutive: int = 2,
) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        for task in build_tasks(sessions, horizon):
            for model_name, predictor in predictors.items():
                try:
                    prediction: object = predictor(task)
                    generation_status = "ok"
                except Exception as error:
                    prediction = None
                    generation_status = f"error:{type(error).__name__}"
                rows.append(
                    score_prediction(
                        task,
                        prediction,
                        cohort=cohort,
                        model_name=model_name,
                        generation_status=generation_status,
                        persistence_min_consecutive=persistence_min_consecutive,
                    )
                )
    return pd.DataFrame(rows)


def aggregate_prediction_scores(
    detailed: pd.DataFrame,
    *,
    eligible_sessions: int,
) -> pd.DataFrame:
    rows = []
    for (cohort, horizon, model_name), group in detailed.groupby(
        ["cohort", "horizon", "model_name"], sort=False
    ):
        scored = group[group["scored"]]
        n_tasks = len(group)
        n_scored = len(scored)
        predicted_blocks = int(scored["predicted_edit_blocks"].sum())
        observed_blocks = int(scored["observed_edit_blocks"].sum())
        matched_credit = float(scored["matched_edit_credit"].sum())
        micro_precision, micro_recall, micro_f1 = prf_from_credit(
            matched_credit, predicted_blocks, observed_blocks
        )
        content_f1 = prf_from_credit(
            float(scored["content_match_credit"].sum()),
            predicted_blocks,
            observed_blocks,
        )[2]
        location_f1 = prf_from_credit(
            float(scored["location_match_credit"].sum()),
            predicted_blocks,
            observed_blocks,
        )[2]
        operation_f1 = prf_from_credit(
            float(scored["operation_match_credit"].sum()),
            predicted_blocks,
            observed_blocks,
        )[2]
        persistence_eligible = scored[scored["persistence_eligible"]]
        persistence_censored = scored[scored["persistence_censored"]]
        persistence_observed_blocks = int(
            persistence_eligible["persistence_observed_blocks"].sum()
        )
        persistence_censored_blocks = int(
            persistence_censored["persistence_censored_blocks"].sum()
        )
        stable_reference_blocks = int(
            persistence_eligible["stable_reference_blocks"].sum()
        )
        stable_predicted_blocks = int(
            persistence_eligible["stable_predicted_blocks"].sum()
        )
        stable_match_credit = float(
            persistence_eligible["stable_match_credit"].sum()
        )
        stable_precision, stable_recall, stable_f1 = prf_from_credit(
            stable_match_credit,
            stable_predicted_blocks,
            stable_reference_blocks,
        )
        rework_eligible = scored[scored["rework_eligible"]]
        reference_change_lines_total = float(
            rework_eligible["reference_change_lines"].sum()
        )
        remaining_rework_lines_total = float(
            rework_eligible["remaining_rework_lines"].sum()
        )
        no_edit_scored = scored[scored["no_edit_target"]]
        reversion_eligible = group[group["reversion_eligible"]]
        reversion_count = int(reversion_eligible["exact_one_step_reversion"].sum())
        rows.append(
            {
                "cohort": cohort,
                "horizon": int(horizon),
                "baseline": model_name,
                "eligible_sessions": eligible_sessions,
                "n_tasks": n_tasks,
                "scored_predictions": n_scored,
                "invalid_predictions": n_tasks - n_scored,
                "prediction_coverage": n_scored / n_tasks if n_tasks else float("nan"),
                "exact_count": int(scored["exact_match"].sum()),
                "exact_match": float(scored["exact_match"].mean()) if n_scored else float("nan"),
                "whole_file_line_jaccard": (
                    float(scored["whole_file_line_jaccard"].mean())
                    if n_scored
                    else float("nan")
                ),
                "no_edit_target_rate": float(group["no_edit_target"].mean()) if n_tasks else float("nan"),
                "false_intervention_eligible_tasks": len(no_edit_scored),
                "false_intervention_count": int(no_edit_scored["false_intervention"].sum()),
                "false_intervention_rate": (
                    float(no_edit_scored["false_intervention"].mean())
                    if len(no_edit_scored)
                    else float("nan")
                ),
                "rework_eligible_tasks": len(rework_eligible),
                "reference_change_lines_total": int(reference_change_lines_total),
                "remaining_rework_lines_total": int(remaining_rework_lines_total),
                "pooled_rework_reduction": (
                    (reference_change_lines_total - remaining_rework_lines_total)
                    / reference_change_lines_total
                    if reference_change_lines_total
                    else float("nan")
                ),
                "mean_reference_change_lines": (
                    float(rework_eligible["reference_change_lines"].mean())
                    if len(rework_eligible)
                    else float("nan")
                ),
                "mean_predicted_change_lines": (
                    float(rework_eligible["predicted_change_lines"].mean())
                    if len(rework_eligible)
                    else float("nan")
                ),
                "mean_remaining_rework_lines": (
                    float(rework_eligible["remaining_rework_lines"].mean())
                    if len(rework_eligible)
                    else float("nan")
                ),
                "mean_rework_reduction": (
                    float(rework_eligible["rework_reduction"].mean())
                    if len(rework_eligible)
                    else float("nan")
                ),
                "rework_improved_rate": (
                    float(rework_eligible["rework_improved"].mean())
                    if len(rework_eligible)
                    else float("nan")
                ),
                "rework_worsened_rate": (
                    float(rework_eligible["rework_worsened"].mean())
                    if len(rework_eligible)
                    else float("nan")
                ),
                "mean_unnecessary_change_lines": (
                    float(no_edit_scored["unnecessary_change_lines"].mean())
                    if len(no_edit_scored)
                    else float("nan")
                ),
                "reversion_eligible_tasks": len(reversion_eligible),
                "reversion_count": reversion_count,
                "reversion_rate": (
                    reversion_count / len(reversion_eligible)
                    if len(reversion_eligible)
                    else float("nan")
                ),
                "reversion_censored_tasks": int(group["reversion_censored"].sum()),
                "immediate_edit_precision": micro_precision,
                "immediate_edit_recall": micro_recall,
                "location_content_edit_f1": micro_f1,
                "content_edit_f1": content_f1,
                "location_edit_f1": location_f1,
                "operation_edit_f1": operation_f1,
                "predicted_edit_blocks": predicted_blocks,
                "observed_edit_blocks": observed_blocks,
                "predicted_edit_rate": float(scored["predicted_edit"].mean()) if n_scored else float("nan"),
                "durability_n": int(scored["durability"].notna().sum()),
                "mean_durability": (
                    float(scored["durability"].mean())
                    if scored["durability"].notna().any()
                    else float("nan")
                ),
                "persistence_min_consecutive": int(
                    group["persistence_min_consecutive"].iloc[0]
                ),
                "persistence_eligible_tasks": len(persistence_eligible),
                "persistence_censored_tasks": len(persistence_censored),
                "persistence_observed_blocks": persistence_observed_blocks,
                "persistence_censored_blocks": persistence_censored_blocks,
                "stable_reference_blocks": stable_reference_blocks,
                "stable_reference_rate": (
                    stable_reference_blocks / persistence_observed_blocks
                    if persistence_observed_blocks
                    else float("nan")
                ),
                "stable_predicted_blocks": stable_predicted_blocks,
                "stable_precision": stable_precision,
                "stable_recall": stable_recall,
                "stable_f1": stable_f1,
            }
        )
    return pd.DataFrame(rows)


def temporary_reward_audit(
    detailed: pd.DataFrame,
    *,
    cohort: str = "strict_no_missing_states",
    horizon: int = 1,
) -> pd.DataFrame:
    """Compare immediate edit credit with credit restricted to stable edits.

    The comparison is made on the same persistence-eligible tasks. Tasks without
    enough follow-up are reported as censored and never treated as unstable.
    """
    required = {
        "cohort",
        "horizon",
        "model_name",
        "scored",
        "persistence_eligible",
        "persistence_censored",
        "observed_edit_blocks",
        "predicted_edit_blocks",
        "persistence_observed_blocks",
        "stable_reference_blocks",
        "stable_predicted_blocks",
        "matched_edit_credit",
        "stable_match_credit",
        "exact_match",
    }
    missing = required - set(detailed.columns)
    if missing:
        raise ValueError(f"Detailed scores are missing columns: {sorted(missing)}")

    data = detailed[
        (detailed["cohort"] == cohort)
        & (detailed["horizon"] == horizon)
    ]
    rows = []
    for model_name, group in data.groupby("model_name", sort=False):
        scored = group[group["scored"]]
        eligible = scored[scored["persistence_eligible"]]
        censored = scored[scored["persistence_censored"]]
        changed = eligible[eligible["observed_edit_blocks"] > 0]
        exact_changed = changed[changed["exact_match"]]

        immediate_credit = float(eligible["matched_edit_credit"].sum())
        stable_credit = float(eligible["stable_match_credit"].sum())
        temporary_credit = max(0.0, immediate_credit - stable_credit)
        observed_blocks = int(eligible["persistence_observed_blocks"].sum())
        predicted_blocks = int(eligible["predicted_edit_blocks"].sum())
        stable_reference_blocks = int(eligible["stable_reference_blocks"].sum())
        stable_predicted_blocks = int(eligible["stable_predicted_blocks"].sum())
        immediate_f1 = prf_from_credit(
            immediate_credit, predicted_blocks, observed_blocks
        )[2]
        stable_f1 = prf_from_credit(
            stable_credit, stable_predicted_blocks, stable_reference_blocks
        )[2]
        exact_with_any_transient = exact_changed[
            exact_changed["stable_reference_blocks"]
            < exact_changed["persistence_observed_blocks"]
        ]
        exact_with_only_transient = exact_changed[
            exact_changed["stable_reference_blocks"] == 0
        ]

        rows.append(
            {
                "cohort": cohort,
                "horizon": horizon,
                "baseline": model_name,
                "persistence_min_consecutive": (
                    int(eligible["persistence_min_consecutive"].iloc[0])
                    if len(eligible)
                    else 2
                ),
                "persistence_eligible_tasks": len(eligible),
                "persistence_censored_tasks": len(censored),
                "eligible_changed_tasks": len(changed),
                "eligible_observed_blocks": observed_blocks,
                "stable_reference_blocks": stable_reference_blocks,
                "immediate_matched_credit": immediate_credit,
                "stable_matched_credit": stable_credit,
                "temporary_credit_gap": temporary_credit,
                "temporary_credit_share": (
                    temporary_credit / immediate_credit
                    if immediate_credit
                    else float("nan")
                ),
                "immediate_location_content_f1": immediate_f1,
                "trajectory_aware_stable_f1": stable_f1,
                "exact_changed_target_matches": len(exact_changed),
                "exact_matches_with_any_transient_reference": len(
                    exact_with_any_transient
                ),
                "exact_matches_with_only_transient_reference": len(
                    exact_with_only_transient
                ),
                "exact_any_transient_rate": (
                    len(exact_with_any_transient) / len(exact_changed)
                    if len(exact_changed)
                    else float("nan")
                ),
                "exact_only_transient_rate": (
                    len(exact_with_only_transient) / len(exact_changed)
                    if len(exact_changed)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def temporary_reward_cases(
    sessions: list[dict],
    detailed: pd.DataFrame,
    *,
    model_name: str = "repeat_last_edit",
    predictor: PredictionFn | None = None,
    cohort: str = "strict_no_missing_states",
    horizon: int = 1,
    limit: int = 8,
) -> pd.DataFrame:
    """Return concrete cases where immediate match credit disappears later."""
    if predictor is None:
        if model_name not in BASELINE_PREDICTORS:
            raise ValueError("A predictor is required for a non-baseline model")
        predictor = BASELINE_PREDICTORS[model_name]

    candidates = detailed[
        (detailed["cohort"] == cohort)
        & (detailed["horizon"] == horizon)
        & (detailed["model_name"] == model_name)
        & detailed["scored"]
        & detailed["persistence_eligible"]
        & (
            detailed["matched_edit_credit"]
            > detailed["stable_match_credit"] + 1e-12
        )
    ].copy()
    candidates["temporary_credit_gap"] = (
        candidates["matched_edit_credit"] - candidates["stable_match_credit"]
    )
    candidates = candidates.sort_values(
        ["temporary_credit_gap", "example_id"], ascending=[False, True]
    ).head(limit)
    task_map = {
        task["example_id"]: task
        for task in build_tasks(sessions, horizon)
    }
    rows = []
    for number, record in enumerate(candidates.to_dict("records"), start=1):
        task = task_map[record["example_id"]]
        prediction = predictor(task)
        follow_up = task["future"][1]
        rows.append(
            {
                "case_id": f"T{number:02d}",
                "example_id": record["example_id"],
                "session_id": record["session_id"],
                "anchor": int(record["anchor"]),
                "immediate_exact_match": bool(record["exact_match"]),
                "observed_edit_blocks": int(record["observed_edit_blocks"]),
                "stable_reference_blocks": int(record["stable_reference_blocks"]),
                "predicted_edit_blocks": int(record["predicted_edit_blocks"]),
                "immediate_matched_credit": float(record["matched_edit_credit"]),
                "stable_matched_credit": float(record["stable_match_credit"]),
                "temporary_credit_gap": float(record["temporary_credit_gap"]),
                "current_to_target_diff": compact_diff(
                    task["current"], task["target"], limit=900
                ),
                "target_to_follow_up_diff": compact_diff(
                    task["target"], follow_up, limit=900
                ),
                "prediction_to_target_diff": compact_diff(
                    prediction, task["target"], limit=900
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(
    sessions: list[dict],
    horizons: tuple[int, ...],
    cohort: str,
    predictors: dict[str, PredictionFn],
    *,
    persistence_min_consecutive: int = 2,
) -> pd.DataFrame:
    summaries = []
    for horizon in horizons:
        detailed = evaluate_predictors_detailed(
            sessions,
            (horizon,),
            cohort,
            predictors,
            persistence_min_consecutive=persistence_min_consecutive,
        )
        summaries.append(
            aggregate_prediction_scores(
                detailed,
                eligible_sessions=len(sessions),
            )
        )
    return pd.concat(summaries, ignore_index=True)


def evaluate_external_predictions(
    sessions: list[dict],
    predictions: pd.DataFrame,
    *,
    cohort: str,
    persistence_min_consecutive: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the shared prediction contract and score external outputs."""
    missing = [
        column
        for column in PREDICTION_REQUIRED_COLUMNS
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if predictions.duplicated(["example_id", "model_name"]).any():
        raise ValueError("Duplicate example_id/model_name prediction rows")
    horizons = tuple(sorted(int(value) for value in predictions["horizon"].unique()))
    task_map = {
        task["example_id"]: task
        for horizon in horizons
        for task in build_tasks(sessions, horizon)
    }
    provided = {}
    for record in predictions.to_dict("records"):
        task = task_map.get(record["example_id"])
        if task is None:
            raise ValueError(f"Unknown example_id: {record['example_id']}")
        if int(record["horizon"]) != task["horizon"]:
            raise ValueError(f"Horizon mismatch: {record['example_id']}")
        provided[(record["example_id"], str(record["model_name"]))] = record

    rows = []
    requested_pairs = sorted(
        {
            (str(record["model_name"]), int(record["horizon"]))
            for record in predictions.to_dict("records")
        }
    )
    for model_name, horizon in requested_pairs:
        for task in build_tasks(sessions, horizon):
            record = provided.get((task["example_id"], model_name))
            if record is None:
                rows.append(
                    score_prediction(
                        task,
                        None,
                        cohort=cohort,
                        model_name=model_name,
                        generation_status="missing",
                        persistence_min_consecutive=persistence_min_consecutive,
                    )
                )
                continue
            rows.append(
                score_prediction(
                    task,
                    record.get("prediction"),
                    cohort=cohort,
                    model_name=model_name,
                    generation_status=str(record["generation_status"]),
                    model_version=str(record.get("model_version", "")),
                    prompt_version=str(record.get("prompt_version", "")),
                    persistence_min_consecutive=persistence_min_consecutive,
                )
            )
    detailed = pd.DataFrame(rows)
    summary = aggregate_prediction_scores(
        detailed,
        eligible_sessions=len(sessions),
    )
    return detailed, summary


def evaluate_baselines(
    sessions: list[dict],
    horizons: tuple[int, ...],
    cohort: str,
) -> pd.DataFrame:
    return evaluate_predictions(
        sessions,
        horizons,
        cohort,
        BASELINE_PREDICTORS,
    )


def headline_results(
    summary: pd.DataFrame,
    *,
    cohort: str = "strict_no_missing_states",
    horizon: int = 1,
) -> pd.DataFrame:
    """Compact presentation table containing only metrics defined for all baselines."""
    columns = [
        "baseline",
        "n_tasks",
        "prediction_coverage",
        "exact_match",
        "content_edit_f1",
        "location_edit_f1",
        "operation_edit_f1",
        "location_content_edit_f1",
        "predicted_edit_rate",
        "false_intervention_rate",
    ]
    return (
        summary[
            (summary["cohort"] == cohort)
            & (summary["horizon"] == horizon)
        ][columns]
        .copy()
        .reset_index(drop=True)
    )


def primary_scorecard(
    summary: pd.DataFrame,
    *,
    cohort: str = "strict_no_missing_states",
    horizon: int = 1,
) -> pd.DataFrame:
    """One decision table spanning coverage, accuracy, harm, work, and stability.

    The metrics deliberately remain separate. A weighted composite would hide
    different failure modes and introduce an arbitrary project preference.
    """
    columns = list(PRIMARY_SCORECARD_COLUMNS)
    scorecard = (
        summary[
            (summary["cohort"] == cohort)
            & (summary["horizon"] == horizon)
        ][columns]
        .copy()
        .reset_index(drop=True)
    )
    required_metrics = columns[2:]
    if scorecard.empty:
        raise ValueError(
            f"No scorecard rows for cohort={cohort!r}, horizon={horizon}."
        )
    if scorecard[required_metrics].isna().any().any():
        missing = scorecard[required_metrics].columns[
            scorecard[required_metrics].isna().any()
        ].tolist()
        raise ValueError(f"Primary scorecard has undefined metrics: {missing}")
    return scorecard


def rework_proxy_results(
    summary: pd.DataFrame,
    *,
    cohort: str = "strict_no_missing_states",
    horizon: int = 1,
) -> pd.DataFrame:
    """Compact offline-work proxy table kept separate from accuracy metrics."""
    columns = [
        "baseline",
        "rework_eligible_tasks",
        "reference_change_lines_total",
        "remaining_rework_lines_total",
        "pooled_rework_reduction",
        "mean_reference_change_lines",
        "mean_predicted_change_lines",
        "mean_remaining_rework_lines",
        "mean_rework_reduction",
        "rework_improved_rate",
        "rework_worsened_rate",
        "mean_unnecessary_change_lines",
    ]
    return (
        summary[
            (summary["cohort"] == cohort)
            & (summary["horizon"] == horizon)
        ][columns]
        .copy()
        .reset_index(drop=True)
    )


def target_type_breakdown(detailed: pd.DataFrame) -> pd.DataFrame:
    """Explain performance separately for insert, delete, replace, and mixed targets."""
    edited = detailed[
        detailed["observed_edit_type"] != "no_edit"
    ].copy()
    rows = []
    for (cohort, horizon, model_name, target_type), group in edited.groupby(
        ["cohort", "horizon", "model_name", "observed_edit_type"],
        sort=False,
    ):
        scored = group[group["scored"]]
        predicted_blocks = int(scored["predicted_edit_blocks"].sum())
        observed_blocks = int(scored["observed_edit_blocks"].sum())
        rows.append(
            {
                "cohort": cohort,
                "horizon": int(horizon),
                "baseline": model_name,
                "target_edit_type": target_type,
                "n_tasks": len(group),
                "prediction_coverage": (
                    len(scored) / len(group) if len(group) else float("nan")
                ),
                "exact_match": (
                    float(scored["exact_match"].mean())
                    if len(scored)
                    else float("nan")
                ),
                "predicted_edit_rate": (
                    float(scored["predicted_edit"].mean())
                    if len(scored)
                    else float("nan")
                ),
                "content_edit_f1": prf_from_credit(
                    float(scored["content_match_credit"].sum()),
                    predicted_blocks,
                    observed_blocks,
                )[2],
                "location_edit_f1": prf_from_credit(
                    float(scored["location_match_credit"].sum()),
                    predicted_blocks,
                    observed_blocks,
                )[2],
                "operation_edit_f1": prf_from_credit(
                    float(scored["operation_match_credit"].sum()),
                    predicted_blocks,
                    observed_blocks,
                )[2],
                "location_content_edit_f1": prf_from_credit(
                    float(scored["matched_edit_credit"].sum()),
                    predicted_blocks,
                    observed_blocks,
                )[2],
            }
        )
    return pd.DataFrame(rows)


def user_grouped_split_manifest(
    sessions: list[dict],
    *,
    seed: int = 20261104,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> pd.DataFrame:
    """Create a deterministic split with no user appearing in two partitions."""
    if train_fraction <= 0 or validation_fraction < 0:
        raise ValueError("Split fractions must be non-negative and train must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test partition.")
    if any(not session.get("user_id") for session in sessions):
        raise ValueError("Every session needs user_id for a leakage-safe split.")

    users = sorted(
        {session["user_id"] for session in sessions},
        key=lambda user_id: hashlib.sha256(
            f"{seed}:{user_id}".encode("utf-8")
        ).hexdigest(),
    )
    train_end = round(len(users) * train_fraction)
    validation_end = train_end + round(len(users) * validation_fraction)
    user_split = {
        user_id: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, user_id in enumerate(users)
    }
    rows = [
        {
            "session_id": session["session_id"],
            "user_id": session["user_id"],
            "source_model": session.get("source_model", ""),
            "outcome_timestamp": session.get("outcome_timestamp", ""),
            "split": user_split[session["user_id"]],
            "split_seed": seed,
        }
        for session in sessions
    ]
    return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)


def evaluation_validity_audit(
    sessions: list[dict],
    predictors: dict[str, PredictionFn] | None = None,
) -> pd.DataFrame:
    """Audit target definition, context ambiguity, and timing sensitivity."""
    predictors = predictors or BASELINE_PREDICTORS
    tasks = list(build_tasks(sessions, 1))
    if not tasks:
        return pd.DataFrame(
            columns=["section", "metric", "count", "denominator", "value"]
        )

    rows: list[dict] = []

    def add(section: str, metric: str, count: float, denominator: float) -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "count": count,
                "denominator": denominator,
                "value": count / denominator if denominator else float("nan"),
            }
        )

    raw_no_change_exact = sum(task["current"] == task["target"] for task in tasks)
    normalized_no_edit = sum(
        not extract_edits(task["current"], task["target"])
        for task in tasks
    )
    whitespace_only = sum(
        task["current"] != task["target"]
        and not extract_edits(task["current"], task["target"])
        for task in tasks
    )
    add("target_definition", "raw_no_change_exact_rate", raw_no_change_exact, len(tasks))
    add("target_definition", "normalized_no_edit_rate", normalized_no_edit, len(tasks))
    add("target_definition", "whitespace_only_target_rate", whitespace_only, len(tasks))

    contexts = pd.DataFrame(
        {
            "previous": [task["previous"] for task in tasks],
            "current": [task["current"] for task in tasks],
            "target": [task["target"] for task in tasks],
            "user_id": [task.get("user_id", "") for task in tasks],
            "session_id": [task["session_id"] for task in tasks],
        }
    )
    context_summary = contexts.groupby("current", sort=False).agg(
        tasks=("target", "size"),
        users=("user_id", "nunique"),
        targets=("target", "nunique"),
    )
    repeated = context_summary[context_summary["tasks"] > 1]
    cross_user = context_summary[context_summary["users"] > 1]
    add(
        "context_ambiguity",
        "tasks_in_repeated_current_states",
        float(repeated["tasks"].sum()),
        len(tasks),
    )
    add(
        "context_ambiguity",
        "tasks_in_cross_user_current_states",
        float(cross_user["tasks"].sum()),
        len(tasks),
    )
    add(
        "context_ambiguity",
        "repeated_contexts_with_multiple_targets",
        float((repeated["targets"] > 1).sum()),
        len(repeated),
    )

    full_context_summary = contexts.groupby(
        ["previous", "current"], sort=False
    ).agg(
        tasks=("target", "size"),
        users=("user_id", "nunique"),
        targets=("target", "nunique"),
    )
    repeated_full = full_context_summary[full_context_summary["tasks"] > 1]
    cross_user_full = full_context_summary[full_context_summary["users"] > 1]
    add(
        "context_ambiguity",
        "tasks_in_repeated_previous_current_contexts",
        float(repeated_full["tasks"].sum()),
        len(tasks),
    )
    add(
        "context_ambiguity",
        "tasks_in_cross_user_previous_current_contexts",
        float(cross_user_full["tasks"].sum()),
        len(tasks),
    )
    add(
        "context_ambiguity",
        "repeated_previous_current_contexts_with_multiple_targets",
        float((repeated_full["targets"] > 1).sum()),
        len(repeated_full),
    )

    for model_name, predictor in predictors.items():
        predictions = [predictor(task) for task in tasks]
        for window in (1, 2, 5):
            exact = sum(
                prediction in task["future"][:window]
                for prediction, task in zip(predictions, tasks)
            )
            add(
                "timing_sensitivity",
                f"{model_name}_exact_within_{window}",
                exact,
                len(tasks),
            )
    return pd.DataFrame(rows)


def edit_size_breakdown(detailed: pd.DataFrame) -> pd.DataFrame:
    """Report baseline behavior by observed line-change magnitude."""
    data = detailed.copy()
    data["target_change_size"] = pd.cut(
        data["reference_change_lines"],
        bins=[-1, 0, 2, 5, 10, np.inf],
        labels=["0", "1-2", "3-5", "6-10", ">10"],
    )
    rows = []
    for (cohort, horizon, model_name, size), group in data.groupby(
        ["cohort", "horizon", "model_name", "target_change_size"],
        observed=True,
        sort=False,
    ):
        scored = group[group["scored"]]
        changed = scored[scored["reference_change_lines"] > 0]
        predicted_blocks = float(scored["predicted_edit_blocks"].sum())
        observed_blocks = float(scored["observed_edit_blocks"].sum())
        rows.append(
            {
                "cohort": cohort,
                "horizon": int(horizon),
                "baseline": model_name,
                "target_change_lines": str(size),
                "n_tasks": len(group),
                "exact_match": float(scored["exact_match"].mean()),
                "location_content_edit_f1": prf_from_credit(
                    float(scored["matched_edit_credit"].sum()),
                    int(predicted_blocks),
                    int(observed_blocks),
                )[2],
                "pooled_rework_reduction": (
                    1
                    - float(changed["remaining_rework_lines"].sum())
                    / float(changed["reference_change_lines"].sum())
                    if len(changed)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def baseline_oracle_ceiling(
    detailed: pd.DataFrame,
    *,
    cohort: str = "strict_no_missing_states",
    horizon: int = 1,
) -> pd.DataFrame:
    """Upper bound from choosing no-change or repeat with target knowledge."""
    selected = detailed[
        (detailed["cohort"] == cohort) & (detailed["horizon"] == horizon)
    ]
    indexed = selected.set_index(["example_id", "model_name"])
    no_change = indexed.xs("no_change", level="model_name").sort_index()
    repeat = indexed.xs("repeat_last_edit", level="model_name").sort_index()
    if not no_change.index.equals(repeat.index):
        raise ValueError("Oracle baselines must cover the same examples.")

    changed = no_change["reference_change_lines"] > 0
    choose_repeat = changed & (
        repeat["remaining_rework_lines"]
        < no_change["remaining_rework_lines"]
    )
    oracle_remaining = no_change["remaining_rework_lines"].copy()
    oracle_remaining.loc[choose_repeat] = repeat.loc[
        choose_repeat, "remaining_rework_lines"
    ]
    exact_union = no_change["exact_match"] | repeat["exact_match"]
    reference_total = float(
        no_change.loc[changed, "reference_change_lines"].sum()
    )
    return pd.DataFrame(
        [
            {
                "cohort": cohort,
                "horizon": horizon,
                "n_tasks": len(no_change),
                "exact_union_rate": float(exact_union.mean()),
                "exact_increment_over_no_change": float(
                    exact_union.mean() - no_change["exact_match"].mean()
                ),
                "repeat_selected_for_rework_tasks": int(choose_repeat.sum()),
                "repeat_selected_rate_changed_targets": float(
                    choose_repeat.sum() / changed.sum()
                ),
                "oracle_pooled_rework_reduction": (
                    1
                    - float(oracle_remaining.loc[changed].sum())
                    / reference_total
                ),
            }
        ]
    )


def _cluster_metric_totals(group: pd.DataFrame) -> dict[str, float]:
    """Sufficient statistics for one trajectory and one model."""
    scored = group[group["scored"]]
    persistence_eligible = scored[scored["persistence_eligible"]]
    rework_eligible = scored[scored["rework_eligible"]]
    return {
        "scored": float(len(scored)),
        "exact": float(scored["exact_match"].sum()),
        "line_jaccard": float(scored["whole_file_line_jaccard"].sum()),
        "matched_credit": float(scored["matched_edit_credit"].sum()),
        "predicted_blocks": float(scored["predicted_edit_blocks"].sum()),
        "observed_blocks": float(scored["observed_edit_blocks"].sum()),
        "no_edit_tasks": float(scored["no_edit_target"].sum()),
        "false_interventions": float(scored["false_intervention"].sum()),
        "rework_eligible_tasks": float(len(rework_eligible)),
        "rework_reduction_sum": float(rework_eligible["rework_reduction"].sum()),
        "reference_change_lines_total": float(
            rework_eligible["reference_change_lines"].sum()
        ),
        "remaining_rework_lines_total": float(
            rework_eligible["remaining_rework_lines"].sum()
        ),
        "stable_match_credit": float(
            persistence_eligible["stable_match_credit"].sum()
        ),
        "stable_predicted_blocks": float(
            persistence_eligible["stable_predicted_blocks"].sum()
        ),
        "stable_reference_blocks": float(
            persistence_eligible["stable_reference_blocks"].sum()
        ),
    }


def _metrics_from_cluster_totals(totals: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Calculate report metrics from resampled trajectory-level totals."""

    def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        output = np.full_like(numerator, np.nan, dtype=float)
        np.divide(numerator, denominator, out=output, where=denominator != 0)
        return output

    precision = safe_divide(totals["matched_credit"], totals["predicted_blocks"])
    recall = safe_divide(totals["matched_credit"], totals["observed_blocks"])
    joint_f1 = np.zeros_like(precision)
    joint_denominator = precision + recall
    valid_joint = np.isfinite(joint_denominator) & (joint_denominator > 0)
    joint_f1[valid_joint] = (
        2 * precision[valid_joint] * recall[valid_joint]
        / joint_denominator[valid_joint]
    )

    stable_precision = safe_divide(
        totals["stable_match_credit"], totals["stable_predicted_blocks"]
    )
    stable_recall = safe_divide(
        totals["stable_match_credit"], totals["stable_reference_blocks"]
    )
    stable_f1 = np.zeros_like(stable_precision)
    stable_denominator = stable_precision + stable_recall
    valid_stable = np.isfinite(stable_denominator) & (stable_denominator > 0)
    stable_f1[valid_stable] = (
        2
        * stable_precision[valid_stable]
        * stable_recall[valid_stable]
        / stable_denominator[valid_stable]
    )

    return {
        "exact_match": safe_divide(totals["exact"], totals["scored"]),
        "whole_file_line_jaccard": safe_divide(
            totals["line_jaccard"], totals["scored"]
        ),
        "location_content_edit_f1": joint_f1,
        "false_intervention_rate": safe_divide(
            totals["false_interventions"], totals["no_edit_tasks"]
        ),
        "mean_rework_reduction": safe_divide(
            totals["rework_reduction_sum"], totals["rework_eligible_tasks"]
        ),
        "pooled_rework_reduction": 1.0 - safe_divide(
            totals["remaining_rework_lines_total"],
            totals["reference_change_lines_total"],
        ),
        "stable_f1": stable_f1,
    }


def paired_cluster_bootstrap(
    detailed: pd.DataFrame,
    model_a: str = "no_change",
    model_b: str = "repeat_last_edit",
    *,
    cohort: str = "strict_no_missing_states",
    horizon: int = 1,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Paired model differences with trajectory-clustered bootstrap intervals.

    The reported delta is model_b minus model_a. Only examples scored by both
    models are retained, and whole trajectories are resampled to avoid treating
    overlapping windows from the same history as independent observations.
    """
    selected = detailed[
        (detailed["cohort"] == cohort)
        & (detailed["horizon"] == horizon)
        & (detailed["model_name"].isin([model_a, model_b]))
    ].copy()
    if selected.empty:
        raise ValueError("No prediction rows match the requested comparison")

    pair_counts = selected.groupby("example_id")["model_name"].nunique()
    paired_ids = pair_counts[pair_counts == 2].index
    selected = selected[selected["example_id"].isin(paired_ids)]
    scored_counts = selected.groupby("example_id")["scored"].sum()
    paired_scored_ids = scored_counts[scored_counts == 2].index
    selected = selected[selected["example_id"].isin(paired_scored_ids)]
    if selected.empty:
        raise ValueError("No examples were scored by both models")

    cluster_rows = []
    for (model_name, session_id), group in selected.groupby(
        ["model_name", "session_id"], sort=True
    ):
        cluster_rows.append(
            {
                "model_name": model_name,
                "session_id": session_id,
                **_cluster_metric_totals(group),
            }
        )
    cluster_frame = pd.DataFrame(cluster_rows)
    sessions_a = set(
        cluster_frame.loc[cluster_frame["model_name"] == model_a, "session_id"]
    )
    sessions_b = set(
        cluster_frame.loc[cluster_frame["model_name"] == model_b, "session_id"]
    )
    common_sessions = sorted(sessions_a & sessions_b)
    if not common_sessions:
        raise ValueError("The models share no scored trajectory clusters")

    statistic_columns = [
        column
        for column in cluster_frame.columns
        if column not in {"model_name", "session_id"}
    ]
    by_model: dict[str, dict[str, np.ndarray]] = {}
    for model_name in [model_a, model_b]:
        indexed = (
            cluster_frame[cluster_frame["model_name"] == model_name]
            .set_index("session_id")
            .reindex(common_sessions)
        )
        by_model[model_name] = {
            column: indexed[column].to_numpy(dtype=float)
            for column in statistic_columns
        }

    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(
        0, len(common_sessions), size=(n_bootstrap, len(common_sessions))
    )

    original_metrics = {}
    bootstrap_metrics = {}
    for model_name in [model_a, model_b]:
        original_totals = {
            column: np.array([values.sum()], dtype=float)
            for column, values in by_model[model_name].items()
        }
        sampled_totals = {
            column: values[draw_indices].sum(axis=1)
            for column, values in by_model[model_name].items()
        }
        original_metrics[model_name] = _metrics_from_cluster_totals(original_totals)
        bootstrap_metrics[model_name] = _metrics_from_cluster_totals(sampled_totals)

    metric_labels = {
        "exact_match": ("Exact match", "higher"),
        "whole_file_line_jaccard": ("Whole-file line Jaccard", "higher"),
        "location_content_edit_f1": ("Location + content F1", "higher"),
        "false_intervention_rate": ("False intervention rate", "lower"),
        "pooled_rework_reduction": ("Pooled rework reduction", "higher"),
        "stable_f1": ("Stable F1", "higher"),
    }
    rows = []
    for metric, (label, preferred_direction) in metric_labels.items():
        point_a = float(original_metrics[model_a][metric][0])
        point_b = float(original_metrics[model_b][metric][0])
        deltas = bootstrap_metrics[model_b][metric] - bootstrap_metrics[model_a][metric]
        valid_deltas = deltas[np.isfinite(deltas)]
        if len(valid_deltas):
            ci_low, ci_high = np.quantile(valid_deltas, [0.025, 0.975])
            probability_positive = float((valid_deltas > 0).mean())
        else:
            ci_low = ci_high = probability_positive = float("nan")
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "preferred_direction": preferred_direction,
                "model_a": model_a,
                "model_b": model_b,
                "model_a_score": point_a,
                "model_b_score": point_b,
                "delta_b_minus_a": point_b - point_a,
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                "bootstrap_probability_delta_gt_zero": probability_positive,
                "n_trajectory_clusters": len(common_sessions),
                "n_paired_tasks": len(paired_scored_ids),
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def persistence_threshold_analysis(
    sessions: list[dict],
    predictors: dict[str, PredictionFn] | None = None,
    *,
    cohort: str = "strict_no_missing_states",
    thresholds: tuple[int, ...] = (2, 3, 5),
) -> pd.DataFrame:
    """Recalculate h=1 stable metrics under several consecutive-state rules."""
    predictors = predictors or BASELINE_PREDICTORS
    rows = []
    keep = [
        "baseline",
        "persistence_min_consecutive",
        "persistence_eligible_tasks",
        "persistence_censored_tasks",
        "persistence_observed_blocks",
        "persistence_censored_blocks",
        "stable_reference_blocks",
        "stable_reference_rate",
        "stable_predicted_blocks",
        "stable_precision",
        "stable_recall",
        "stable_f1",
    ]
    for threshold in thresholds:
        detailed = evaluate_predictors_detailed(
            sessions,
            (1,),
            cohort,
            predictors,
            persistence_min_consecutive=threshold,
        )
        summary = aggregate_prediction_scores(
            detailed,
            eligible_sessions=len(sessions),
        )
        rows.append(summary[keep])
    return pd.concat(rows, ignore_index=True)


def synthetic_stress_cases() -> pd.DataFrame:
    current = "def f(x):\n    value = x + 1\n    return value\n"
    exact = "def f(x):\n    value = x + 1\n    print(value)\n    return value\n"
    wrong_location = "def f(x):\n    print(value)\n    value = x + 1\n    return value\n"
    partial = "def f(x):\n    value = x + 1\n    print(x)\n    return value\n"
    second_edit = "def f(x):\n    value = x + 2\n    print(value)\n    return value\n"
    deleted = "def f(x):\n    return value\n"

    cases = [
        ("exact_edit", exact, exact, [exact, exact]),
        ("correct_content_wrong_location", wrong_location, exact, [exact, exact]),
        ("partial_content", partial, exact, [exact, exact]),
        ("stable_prediction", exact, exact, [exact, second_edit]),
        ("reverted_prediction", exact, exact, [exact, current]),
        ("missed_second_edit", exact, second_edit, [second_edit, second_edit]),
        ("deletion", deleted, deleted, [deleted, deleted]),
    ]
    rows = []
    for name, prediction, target, future in cases:
        predicted_edits = extract_edits(current, prediction)
        observed_edits = extract_edits(current, target)
        precision, recall, exact_location_f1 = soft_edit_prf(
            predicted_edits,
            observed_edits,
        )
        tolerant_location_f1 = soft_edit_prf(
            predicted_edits,
            observed_edits,
            tolerant=True,
        )[2]
        content_f1 = prf_from_credit(
            optimal_match_credit(
                predicted_edits, observed_edits, payload_similarity
            ),
            len(predicted_edits),
            len(observed_edits),
        )[2]
        location_f1 = prf_from_credit(
            optimal_match_credit(
                predicted_edits, observed_edits, location_similarity
            ),
            len(predicted_edits),
            len(observed_edits),
        )[2]
        operation_f1 = prf_from_credit(
            optimal_match_credit(
                predicted_edits, observed_edits, operation_similarity
            ),
            len(predicted_edits),
            len(observed_edits),
        )[2]
        rows.append(
            {
                "case": name,
                "exact_match": prediction == target,
                "whole_file_line_jaccard": whole_file_line_jaccard(prediction, target),
                "location_content_precision": precision,
                "location_content_recall": recall,
                "content_edit_f1": content_f1,
                "location_edit_f1": location_f1,
                "operation_edit_f1": operation_f1,
                "exact_location_content_f1": exact_location_f1,
                "tolerant_location_content_f1": tolerant_location_f1,
                "durability_retention": durability_score(current, prediction, future),
            }
        )
    return pd.DataFrame(rows)


def select_case_studies(sessions: list[dict], limit_each: int = 5) -> pd.DataFrame:
    buckets: dict[str, list[dict]] = {
        "duplicate_next_state": [],
        "high_overlap_but_not_exact": [],
        "one_step_reversion": [],
        "repeat_last_edit_helps": [],
        "repeat_last_edit_hurts": [],
    }
    for session in sessions:
        states = session["states"]
        for anchor in range(1, len(states) - 1):
            previous, current, target = states[anchor - 1], states[anchor], states[anchor + 1]
            repeat_prediction = repeat_last_edit(previous, current)
            no_change_score = whole_file_line_jaccard(current, target)
            repeat_score = whole_file_line_jaccard(repeat_prediction, target)
            base = {
                "session_id": session["session_id"],
                "anchor": anchor,
                "current_lines": len(current.splitlines()),
                "target_lines": len(target.splitlines()),
                "no_change_line_jaccard": no_change_score,
                "repeat_line_jaccard": repeat_score,
            }
            if current == target and len(buckets["duplicate_next_state"]) < limit_each:
                buckets["duplicate_next_state"].append(base)
            if current != target and no_change_score >= 0.8 and len(buckets["high_overlap_but_not_exact"]) < limit_each:
                buckets["high_overlap_but_not_exact"].append(base)
            if anchor + 2 < len(states) and current != target and current == states[anchor + 2] and len(buckets["one_step_reversion"]) < limit_each:
                buckets["one_step_reversion"].append(base)
            if repeat_score - no_change_score >= 0.2 and len(buckets["repeat_last_edit_helps"]) < limit_each:
                buckets["repeat_last_edit_helps"].append(base)
            if no_change_score - repeat_score >= 0.2 and len(buckets["repeat_last_edit_hurts"]) < limit_each:
                buckets["repeat_last_edit_hurts"].append(base)

        if all(len(rows) >= limit_each for rows in buckets.values()):
            break

    output = []
    for pattern, rows in buckets.items():
        output.extend({"pattern": pattern, **row} for row in rows)
    return pd.DataFrame(output)


def compact_diff(source: str, destination: str, limit: int = 1200) -> str:
    """Short line diff for a human-review CSV cell."""
    text = "".join(
        unified_diff(
            source.splitlines(keepends=True),
            destination.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
            n=2,
        )
    )
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def build_case_review_queue(
    sessions: list[dict],
    *,
    cases_per_pattern: int = 4,
) -> pd.DataFrame:
    """Create a balanced 20-case queue with evidence and blank review fields."""
    candidates = select_case_studies(sessions, limit_each=cases_per_pattern)
    session_map = {session["session_id"]: session["states"] for session in sessions}
    rows = []
    for number, case in enumerate(candidates.to_dict("records"), start=1):
        states = session_map[case["session_id"]]
        anchor = int(case["anchor"])
        previous = states[anchor - 1]
        current = states[anchor]
        target = states[anchor + 1]
        next_state = states[anchor + 2] if anchor + 2 < len(states) else None
        repeat_prediction = repeat_last_edit(previous, current)
        rows.append(
            {
                "case_id": f"C{number:02d}",
                "pattern": case["pattern"],
                "session_id": case["session_id"],
                "anchor": anchor,
                "previous_to_current_diff": compact_diff(previous, current),
                "current_to_target_diff": compact_diff(current, target),
                "target_to_next_diff": (
                    compact_diff(target, next_state) if next_state is not None else ""
                ),
                "repeat_prediction_to_target_diff": compact_diff(
                    repeat_prediction, target
                ),
                "auto_exact_reversion": bool(
                    next_state is not None
                    and target != current
                    and next_state == current
                ),
                "review_status": "pending",
                "manual_label": "",
                "reviewer_notes": "",
            }
        )
    return pd.DataFrame(rows)


def preserve_case_reviews(queue: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Keep completed manual decisions when the automated evidence is rebuilt."""
    output = queue.copy()
    for decision in existing.to_dict("records"):
        if decision.get("review_status") not in {"reviewed", "exclude"}:
            continue
        match = output["case_id"] == decision["case_id"]
        for column in ["review_status", "manual_label", "reviewer_notes"]:
            output.loc[match, column] = decision.get(column, "")
    return output


def apply_fixed_case_audit(review: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen first-pass taxonomy and reconstructability decisions."""
    unknown = set(review["case_id"]) - set(CASE_AUDIT_DECISIONS)
    missing = set(CASE_AUDIT_DECISIONS) - set(review["case_id"])
    if unknown or missing:
        raise ValueError(
            f"Case audit mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    output = review.copy()
    for index, row in output.iterrows():
        label, language, syntax_status = CASE_AUDIT_DECISIONS[row["case_id"]]
        output.loc[index, "review_status"] = (
            "exclude" if label == "formatting_only_exclude" else "reviewed"
        )
        output.loc[index, "manual_label"] = label
        output.loc[index, "taxonomy_definition"] = ERROR_TAXONOMY[label]
        output.loc[index, "language_hint"] = language
        output.loc[index, "syntax_status"] = syntax_status
        output.loc[index, "text_replay_ready"] = True
        output.loc[index, "execution_ready"] = False
        output.loc[index, "reconstructability_tier"] = (
            "syntax_only"
            if syntax_status == "standalone_parse_pass"
            else "text_only"
        )
        output.loc[index, "execution_blocker"] = (
            "No filename, complete project, dependency environment, tests, or entrypoint."
        )
    return output


def case_audit_summary(review: pd.DataFrame) -> pd.DataFrame:
    """Compact denominators for the fixed case and reconstructability audit."""
    total = len(review)
    rows = []

    def add(section: str, metric: str, count: int, denominator: int, meaning: str) -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "count": count,
                "denominator": denominator,
                "rate": count / denominator if denominator else float("nan"),
                "interpretation": meaning,
            }
        )

    add(
        "review_status",
        "retained_cases",
        int((review["review_status"] == "reviewed").sum()),
        total,
        "Cases retained for qualitative evidence.",
    )
    add(
        "review_status",
        "formatting_only_exclusions",
        int((review["review_status"] == "exclude").sum()),
        total,
        "Formatting-only cases excluded from semantic interpretation.",
    )
    add(
        "reconstructability",
        "text_replay_ready",
        int(review["text_replay_ready"].sum()),
        total,
        "Full recorded states support deterministic text-diff replay.",
    )
    add(
        "reconstructability",
        "language_identified",
        int((review["language_hint"] != "unknown").sum()),
        total,
        "Language could be identified conservatively from the snippet.",
    )
    add(
        "reconstructability",
        "standalone_syntax_pass",
        int((review["syntax_status"] == "standalone_parse_pass").sum()),
        total,
        "Target passes a standalone Python AST or Node syntax check.",
    )
    add(
        "reconstructability",
        "fragment_or_parse_fail",
        int((review["syntax_status"] == "fragment_or_parse_fail").sum()),
        total,
        "Failure cannot separate invalid code from an incomplete snippet.",
    )
    add(
        "reconstructability",
        "execution_ready",
        int(review["execution_ready"].sum()),
        total,
        "Requires project context, dependencies, tests, and an entrypoint.",
    )
    for label in ERROR_TAXONOMY:
        count = int((review["manual_label"] == label).sum())
        if count:
            add(
                "taxonomy",
                label,
                count,
                total,
                ERROR_TAXONOMY[label],
            )
    return pd.DataFrame(rows)


def source_model_heterogeneity(
    detailed: pd.DataFrame,
    *,
    cohort: str = "strict_no_missing_states",
    horizon: int = 1,
    min_tasks: int = 300,
    min_trajectories: int = 40,
    min_users: int = 30,
) -> pd.DataFrame:
    """Descriptive strata by the model that produced the source completion."""
    required = {"source_model", "user_id", "session_id"}
    if not required.issubset(detailed.columns):
        raise ValueError(f"Detailed scores are missing metadata: {sorted(required - set(detailed.columns))}")
    data = detailed[
        (detailed["cohort"] == cohort) & (detailed["horizon"] == horizon)
    ]
    rows = []
    for source_model, group in data.groupby("source_model", sort=False):
        no_change = group[group["model_name"] == "no_change"]
        repeat = group[group["model_name"] == "repeat_last_edit"]
        changed = repeat[repeat["reference_change_lines"] > 0]
        tasks = len(no_change)
        trajectories = int(no_change["session_id"].nunique())
        users = int(no_change["user_id"].nunique())
        included = (
            tasks >= min_tasks
            and trajectories >= min_trajectories
            and users >= min_users
        )
        rows.append(
            {
                "source_model": source_model,
                "n_tasks": tasks,
                "n_trajectories": trajectories,
                "n_users": users,
                "included_in_descriptive_comparison": included,
                "normalized_no_edit_rate": float(no_change["no_edit_target"].mean()),
                "median_target_change_lines": float(no_change["reference_change_lines"].median()),
                "repeat_exact_match": float(repeat["exact_match"].mean()),
                "repeat_location_content_f1": prf_from_credit(
                    float(repeat["matched_edit_credit"].sum()),
                    int(repeat["predicted_edit_blocks"].sum()),
                    int(repeat["observed_edit_blocks"].sum()),
                )[2],
                "repeat_false_intervention_rate": (
                    float(repeat["false_intervention"].sum())
                    / float(repeat["no_edit_target"].sum())
                    if repeat["no_edit_target"].sum()
                    else float("nan")
                ),
                "repeat_pooled_rework_reduction": (
                    1
                    - float(changed["remaining_rework_lines"].sum())
                    / float(changed["reference_change_lines"].sum())
                    if len(changed)
                    else float("nan")
                ),
                "descriptive_threshold": (
                    f">={min_tasks} tasks; >={min_trajectories} trajectories; >={min_users} users"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("n_tasks", ascending=False).reset_index(drop=True)


def metric_claim_specification() -> pd.DataFrame:
    """Frozen metric definitions and the claim boundary for each one."""
    columns = [
        "metric",
        "research_role",
        "inputs",
        "calculation",
        "denominator",
        "censoring_or_missing_rule",
        "known_failure_case",
        "supported_conclusion",
        "requires_additional_evidence",
    ]
    rows = [
        (
            "prediction_coverage",
            "Output reliability",
            "Requested examples and scoreable model outputs",
            "scoreable outputs / requested examples",
            "All requested examples",
            "Missing, invalid, or failed outputs remain in the denominator",
            "High coverage does not imply accurate predictions",
            "Whether the system returned a scoreable prediction consistently",
            "Correctness, usefulness, or latency acceptability",
        ),
        (
            "exact_match",
            "Immediate full-state match",
            "Predicted full text and observed target text",
            "Count of byte-for-byte equal predictions and targets / scoreable outputs",
            "Scoreable predictions",
            "Failed outputs reduce coverage and are not silently added to the accuracy denominator",
            "One observed target is not the only potentially valid edit; no-edit cases can dominate",
            "Whether the predicted file exactly matches the recorded state at the chosen horizon",
            "Semantic correctness, intent, or whether an alternative edit was valid",
        ),
        (
            "exact_match_within_future_window",
            "Later-state timing sensitivity",
            "Prediction and the next k recorded states",
            "Tasks where prediction equals any state in the fixed future window / all tasks",
            "All tasks at the anchor; k must be reported",
            "Only available recorded states are searched; no missing state is imputed",
            "A later match may be coincidence and does not prove the edit was useful when suggested",
            "Whether strict next-state mismatch is plausibly explained by recorded timing",
            "Causal timing error, developer acceptance, or time saved",
        ),
        (
            "location_content_edit_f1",
            "Immediate next-edit match",
            "Line edit blocks from current-to-prediction and current-to-target",
            "Global one-to-one location-and-content credit, converted to micro precision/recall/F1",
            "All predicted and observed edit blocks on scoreable tasks",
            "No-edit tasks contain no positive reference blocks; abstention is evaluated separately",
            "Partial similarity can reward a plausible but semantically wrong edit",
            "Whether the model recovers the observed edit content near the observed location",
            "Semantic correctness, executability, or developer preference",
        ),
        (
            "content_location_operation_f1_diagnostics",
            "Failure decomposition",
            "The same predicted and observed line edit blocks",
            "Separate micro F1 scores using payload, source location, or edit operation credit",
            "All predicted and observed edit blocks on scoreable tasks",
            "No positive edit class yields no recall denominator",
            "A high component score can coexist with a low joint score",
            "Whether failure is primarily content, placement, or insert/delete/replace mismatch",
            "That any one component alone makes an edit correct",
        ),
        (
            "false_intervention_rate",
            "Unnecessary intervention",
            "Observed no-edit targets and predicted edit blocks",
            "No-edit targets receiving at least one predicted edit / observed no-edit targets",
            "Scoreable tasks with no observed edit block",
            "Whitespace-normalized no-edit is fixed by protocol",
            "An unrecorded but useful proactive edit is still counted as intervention",
            "How often the model changes code when the recorded next state does not",
            "Whether every intervention was harmful to the developer",
        ),
        (
            "pooled_rework_reduction",
            "Remaining text work",
            "Line costs current-to-target and prediction-to-target",
            "(total required line cost - total remaining line cost) / total required line cost",
            "Total current-to-target line cost over changed targets",
            "No-edit targets are reported separately as unnecessary change lines",
            "Line count is insensitive to semantic difficulty and developer time",
            "Whether prediction moves recorded targets closer under a transparent text proxy",
            "Actual time saved, semantic correctness, or production utility",
        ),
        (
            "mean_durability",
            "Retention of the prediction",
            "Predicted edit payload and up to five later recorded states",
            "Mean weighted near-location retention of predicted payloads",
            "Tasks with a non-empty predicted edit and at least one later observation",
            "No predicted edit has no durability score; shorter follow-up contributes fewer states",
            "Similar text can survive for reasons unrelated to the predicted edit",
            "Whether predicted text remains visible near its original location",
            "That persistence means correctness or usefulness",
        ),
        (
            "reference_reversion_rate",
            "Observed strict reversion",
            "Current, edited target, and the immediately following recorded state",
            "Edited targets whose next state exactly returns to current / eligible edited targets",
            "Horizon-1 edited targets with one follow-up state",
            "Edited targets without follow-up are censored, not labeled non-reverting",
            "Misses partial, moved, or semantic reversions",
            "Frequency of exact one-step return in the observed reference trajectory",
            "A model quality claim; this is a dataset diagnostic",
        ),
        (
            "stable_f1",
            "Trajectory-aware durable edit match",
            "Prediction, observed target edits, and the required consecutive future states",
            "Location-and-content F1 after restricting reference blocks to those that persist",
            "Predicted blocks and stable reference blocks on persistence-eligible tasks",
            "Insufficient follow-up is censored; two states is primary, three and five are sensitivity checks",
            "The persistence rule can miss refactors or semantically retained changes that move",
            "Whether predicted edits match observed edits that remain under the fixed rule",
            "Semantic correctness, causal usefulness, or developer intent",
        ),
        (
            "temporary_credit_share",
            "Immediate-versus-durable disagreement",
            "Immediate and stable location-content match credit on the same eligible tasks",
            "(immediate matched credit - stable matched credit) / immediate matched credit",
            "Immediate matched credit on persistence-eligible tasks",
            "Undefined when a model earns zero immediate credit; censored tasks are excluded",
            "Depends on the fixed persistence rule and is not a causal label of abandoned intent",
            "How much immediate edit-match credit is unavailable after requiring observed persistence",
            "Why the edit disappeared or whether the developer rejected it",
        ),
    ]
    return pd.DataFrame(rows, columns=columns)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(args.data)
    sessions, drop_none_sessions, audit = load_sessions(frame)
    baselines = pd.concat(
        [
            evaluate_baselines(sessions, (1, 2, 5), "strict_no_missing_states"),
            evaluate_baselines(drop_none_sessions, (1, 2, 5), "drop_none_states"),
        ],
        ignore_index=True,
    )
    stress_cases = synthetic_stress_cases()
    repeat_audit = repeat_activity_audit(sessions)
    prediction_level_h1 = evaluate_predictors_detailed(
        sessions,
        (1,),
        "strict_no_missing_states",
        BASELINE_PREDICTORS,
    )
    headline = headline_results(baselines)
    scorecard = primary_scorecard(baselines)
    rework_results = rework_proxy_results(baselines)
    operation_breakdown = target_type_breakdown(prediction_level_h1)
    size_breakdown = edit_size_breakdown(prediction_level_h1)
    validity_audit = evaluation_validity_audit(sessions)
    oracle_ceiling = baseline_oracle_ceiling(prediction_level_h1)
    split_manifest = user_grouped_split_manifest(sessions)
    persistence_analysis = persistence_threshold_analysis(sessions)
    paired_comparison = paired_cluster_bootstrap(prediction_level_h1)
    review_queue = build_case_review_queue(sessions)
    review_path = args.output / "case_review_20.csv"
    if review_path.exists():
        review_queue = preserve_case_reviews(
            review_queue, pd.read_csv(review_path)
        )
    review_queue = apply_fixed_case_audit(review_queue)
    review_summary = case_audit_summary(review_queue)
    heterogeneity = source_model_heterogeneity(prediction_level_h1)
    metric_spec = metric_claim_specification()
    temporary_audit = temporary_reward_audit(prediction_level_h1)
    temporary_cases = temporary_reward_cases(
        sessions, prediction_level_h1, limit=8
    )
    protocol = protocol_specification()

    (args.output / "data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    baselines.to_csv(args.output / "baseline_full.csv", index=False)
    stress_cases.to_csv(args.output / "metric_stress_cases.csv", index=False)
    repeat_audit.to_csv(args.output / "repeat_activity_audit.csv", index=False)
    prediction_level_h1.to_parquet(
        args.output / "prediction_level_h1.parquet", index=False
    )
    headline.to_csv(args.output / "headline_results.csv", index=False)
    scorecard.to_csv(args.output / "primary_scorecard.csv", index=False)
    rework_results.to_csv(args.output / "rework_proxy_results.csv", index=False)
    operation_breakdown.to_csv(
        args.output / "target_type_breakdown.csv", index=False
    )
    size_breakdown.to_csv(
        args.output / "edit_size_breakdown.csv", index=False
    )
    validity_audit.to_csv(
        args.output / "evaluation_validity_audit.csv", index=False
    )
    oracle_ceiling.to_csv(
        args.output / "baseline_oracle_ceiling.csv", index=False
    )
    split_manifest.to_csv(
        args.output / "user_grouped_split_manifest.csv", index=False
    )
    persistence_analysis.to_csv(
        args.output / "persistence_thresholds.csv", index=False
    )
    paired_comparison.to_csv(
        args.output / "paired_baseline_comparison.csv", index=False
    )
    pd.DataFrame(
        columns=[
            *PREDICTION_REQUIRED_COLUMNS,
            "model_version",
            "prompt_version",
        ]
    ).to_csv(args.output / "prediction_contract_template.csv", index=False)
    review_queue.to_csv(args.output / "case_review_20.csv", index=False)
    review_summary.to_csv(args.output / "case_audit_summary.csv", index=False)
    heterogeneity.to_csv(
        args.output / "source_model_heterogeneity.csv", index=False
    )
    metric_spec.to_csv(
        args.output / "metric_claim_specification.csv", index=False
    )
    temporary_audit.to_csv(
        args.output / "temporary_reward_audit.csv", index=False
    )
    temporary_cases.to_csv(
        args.output / "temporary_reward_cases.csv", index=False
    )
    (args.output / "evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    print(json.dumps(audit, indent=2))
    print("\nBASELINES\n", baselines.to_string(index=False))
    print("\nSTRESS CASES\n", stress_cases.to_string(index=False))
    print("\nREPEAT ACTIVITY\n", repeat_audit.to_string(index=False))
    print("\nHEADLINE RESULTS\n", headline.to_string(index=False))
    print("\nREWORK PROXY\n", rework_results.to_string(index=False))
    print("\nPERSISTENCE THRESHOLDS\n", persistence_analysis.to_string(index=False))
    print("\nPAIRED CLUSTER BOOTSTRAP\n", paired_comparison.to_string(index=False))
    print("\nCASE REVIEW STATUS\n", review_queue["review_status"].value_counts().to_string())
    print("\nTEMPORARY REWARD AUDIT\n", temporary_audit.to_string(index=False))


if __name__ == "__main__":
    main()
