from __future__ import annotations

import hashlib
from difflib import unified_diff
from statistics import mean
import numpy as np
import pandas as pd
from .constants import ERROR_TAXONOMY, CASE_AUDIT_DECISIONS, PRIMARY_SCORECARD_COLUMNS
from .edits import (extract_edits, whole_file_line_jaccard, prf_from_credit, durability_score, location_similarity, operation_similarity, optimal_match_credit, payload_similarity, soft_edit_prf)
from .predictors import BASELINE_PREDICTORS, repeat_last_edit
from .tasks import build_tasks
from .schemas import Session
from .metrics import aggregate_prediction_scores
from .runner import evaluate_predictors_detailed


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
    sessions = [s.to_legacy() if isinstance(s, Session) else s for s in sessions]
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
    sessions = [s.to_legacy() if isinstance(s, Session) else s for s in sessions]
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
    sessions = [s.to_legacy() if isinstance(s, Session) else s for s in sessions]
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

