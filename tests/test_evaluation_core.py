import math
import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legacy"))

from evaluation_core_original import (  # noqa: E402
    CASE_AUDIT_DECISIONS,
    ERROR_TAXONOMY,
    PREDICTION_REQUIRED_COLUMNS,
    PRIMARY_SCORECARD_COLUMNS,
    aggregate_prediction_scores,
    apply_fixed_case_audit,
    build_model_inputs,
    build_tasks,
    evaluate_external_predictions,
    evaluate_predictors_detailed,
    evaluation_validity_audit,
    greedy_match_credit,
    line_edit_cost,
    metric_claim_specification,
    optimal_match_credit,
    paired_cluster_bootstrap,
    persistence_threshold_analysis,
    primary_scorecard,
    protocol_specification,
    score_prediction,
    temporary_reward_audit,
    temporary_reward_cases,
    user_grouped_split_manifest,
)


def first_task(states, horizon=1):
    sessions = [{"session_id": "s1", "states": states}]
    return next(build_tasks(sessions, horizon))


class EvaluationCoreTests(unittest.TestCase):
    def test_metric_claim_specification_covers_tracker_requirements(self):
        specification = metric_claim_specification().set_index("metric")
        required = {
            "exact_match",
            "exact_match_within_future_window",
            "location_content_edit_f1",
            "false_intervention_rate",
            "pooled_rework_reduction",
            "mean_durability",
            "reference_reversion_rate",
            "stable_f1",
            "temporary_credit_share",
        }
        self.assertTrue(required.issubset(specification.index))
        self.assertFalse(specification.isna().any().any())
        self.assertTrue(
            specification["denominator"].str.len().gt(0).all()
        )
        self.assertTrue(
            specification["requires_additional_evidence"].str.len().gt(0).all()
        )

    def test_temporary_reward_audit_separates_immediate_and_stable_credit(self):
        sessions = [
            {
                "session_id": "stable",
                "states": ["old\n", "base\n", "base\nx\n", "base\nx\n"],
            },
            {
                "session_id": "temporary",
                "states": ["old\n", "base\n", "base\nx\n", "base\n"],
            },
        ]
        detailed = evaluate_predictors_detailed(
            sessions,
            (1,),
            "test",
            {"copy_target": lambda task: task["target"]},
        )
        audit = temporary_reward_audit(
            detailed, cohort="test", horizon=1
        ).iloc[0]
        self.assertEqual(audit["persistence_eligible_tasks"], 2)
        self.assertEqual(audit["immediate_matched_credit"], 2.0)
        self.assertEqual(audit["stable_matched_credit"], 1.0)
        self.assertEqual(audit["temporary_credit_share"], 0.5)
        self.assertEqual(audit["exact_changed_target_matches"], 2)
        self.assertEqual(
            audit["exact_matches_with_only_transient_reference"], 1
        )

        cases = temporary_reward_cases(
            sessions,
            detailed,
            model_name="copy_target",
            predictor=lambda task: task["target"],
            cohort="test",
            limit=5,
        )
        self.assertEqual(len(cases), 1)
        self.assertTrue(cases.iloc[0]["immediate_exact_match"])
        self.assertEqual(cases.iloc[0]["stable_matched_credit"], 0.0)

    def test_fixed_case_audit_is_complete_and_execution_conservative(self):
        review = pd.DataFrame(
            {
                "case_id": list(CASE_AUDIT_DECISIONS),
                "review_status": "",
                "manual_label": "",
                "reviewer_notes": "",
            }
        )
        audited = apply_fixed_case_audit(review)
        self.assertEqual(len(audited), 20)
        self.assertTrue(set(audited["manual_label"]).issubset(ERROR_TAXONOMY))
        self.assertEqual(int(audited["execution_ready"].sum()), 0)
        self.assertEqual(
            int((audited["syntax_status"] == "standalone_parse_pass").sum()),
            3,
        )

    def test_protocol_freeze_matches_scorecard_and_prediction_contract(self):
        protocol = protocol_specification()
        self.assertEqual(
            protocol["primary_scorecard_columns"],
            list(PRIMARY_SCORECARD_COLUMNS),
        )
        self.assertEqual(
            protocol["prediction_required_columns"],
            list(PREDICTION_REQUIRED_COLUMNS),
        )
        self.assertEqual(protocol["split"]["unit"], "user_id")
        self.assertEqual(
            protocol["edit_matching"],
            "global_one_to_one_linear_assignment",
        )

    def test_optimal_assignment_can_beat_greedy_matching(self):
        matrix = [[0.9, 0.8], [0.8, 0.0]]
        scorer = lambda predicted, observed: matrix[predicted][observed]
        self.assertEqual(greedy_match_credit([0, 1], [0, 1], scorer), 0.9)
        self.assertEqual(optimal_match_credit([0, 1], [0, 1], scorer), 1.6)

    def test_user_grouped_split_has_no_user_overlap_and_is_deterministic(self):
        sessions = [
            {
                "session_id": f"s{index}",
                "user_id": f"u{index % 10}",
                "states": ["a", "b", "c"],
            }
            for index in range(20)
        ]
        first = user_grouped_split_manifest(sessions, seed=7)
        second = user_grouped_split_manifest(sessions, seed=7)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue((first.groupby("user_id")["split"].nunique() == 1).all())
        self.assertEqual(set(first["split"]), {"train", "validation", "test"})

    def test_validity_audit_separates_raw_and_normalized_no_edit(self):
        sessions = [
            {
                "session_id": "s1",
                "user_id": "u1",
                "states": ["a", "x ", "x", "y"],
            }
        ]
        audit = evaluation_validity_audit(
            sessions,
            {"no_change": lambda task: task["current"]},
        ).set_index("metric")
        self.assertEqual(audit.loc["raw_no_change_exact_rate", "count"], 0)
        self.assertEqual(audit.loc["normalized_no_edit_rate", "count"], 1)
        self.assertEqual(audit.loc["whitespace_only_target_rate", "count"], 1)

    def test_model_inputs_do_not_leak_target_or_future(self):
        sessions = [{"session_id": "s1", "states": ["a", "b", "c"]}]
        inputs = build_model_inputs(sessions, (1,))
        self.assertEqual(len(inputs), 1)
        self.assertNotIn("target", inputs.columns)
        self.assertNotIn("future", inputs.columns)
        self.assertEqual(inputs.iloc[0]["example_id"], "s1::a1::h1")

    def test_exact_edit_receives_full_edit_credit(self):
        current = "a\nb\n"
        target = "a\nx\nb\n"
        task = first_task(["a\n", current, target, target])
        result = score_prediction(
            task, target, cohort="test", model_name="model"
        )
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["edit_precision"], 1.0)
        self.assertEqual(result["edit_recall"], 1.0)
        self.assertEqual(result["edit_f1"], 1.0)

    def test_right_content_at_wrong_location_gets_no_location_credit(self):
        current = "a\nb\nc\n"
        target = "a\nb\nx\nc\n"
        prediction = "a\nx\nb\nc\n"
        task = first_task(["a\n", current, target, target])
        result = score_prediction(
            task, prediction, cohort="test", model_name="model"
        )
        self.assertEqual(result["matched_edit_credit"], 0.0)
        self.assertEqual(result["edit_f1"], 0.0)
        self.assertEqual(result["content_edit_f1"], 1.0)
        self.assertEqual(result["location_edit_f1"], 0.0)
        self.assertEqual(result["operation_edit_f1"], 1.0)

    def test_operation_is_scored_separately_from_content_and_location(self):
        current = "a\nb\n"
        target = "a\nx\nb\n"
        prediction = "a\nx\n"
        task = first_task(["a\n", current, target, target])
        result = score_prediction(
            task, prediction, cohort="test", model_name="model"
        )
        self.assertEqual(result["content_edit_f1"], 1.0)
        self.assertEqual(result["location_edit_f1"], 1.0)
        self.assertEqual(result["operation_edit_f1"], 0.0)
        self.assertEqual(result["edit_f1"], 1.0)

    def test_edit_on_no_edit_target_is_false_intervention(self):
        task = first_task(["a\n", "a\n", "a\n", "a\n"])
        result = score_prediction(
            task, "a\nb\n", cohort="test", model_name="model"
        )
        self.assertTrue(result["no_edit_target"])
        self.assertTrue(result["predicted_edit"])
        self.assertTrue(result["false_intervention"])
        self.assertTrue(math.isnan(result["edit_recall"]))

    def test_no_change_has_zero_joint_f1_when_target_contains_an_edit(self):
        sessions = [{"session_id": "s1", "states": ["a", "b", "c"]}]
        detailed = evaluate_predictors_detailed(
            sessions,
            (1,),
            "test",
            {"no_change": lambda task: task["current"]},
        )
        summary = aggregate_prediction_scores(detailed, eligible_sessions=1)
        self.assertEqual(summary.iloc[0]["location_content_edit_f1"], 0.0)

    def test_primary_scorecard_is_compact_and_fully_defined(self):
        sessions = [
            {"session_id": "s1", "states": ["a", "a", "a", "a"]},
            {"session_id": "s2", "states": ["a", "b", "c", "c"]},
        ]
        detailed = evaluate_predictors_detailed(
            sessions,
            (1,),
            "test",
            {
                "no_change": lambda task: task["current"],
                "copy_target": lambda task: task["target"],
            },
        )
        summary = aggregate_prediction_scores(detailed, eligible_sessions=2)
        scorecard = primary_scorecard(summary, cohort="test")
        self.assertEqual(
            scorecard.columns.tolist(),
            [
                "baseline",
                "n_tasks",
                "prediction_coverage",
                "exact_match",
                "location_content_edit_f1",
                "false_intervention_rate",
                "pooled_rework_reduction",
                "stable_f1",
            ],
        )
        self.assertFalse(scorecard.isna().any().any())

    def test_rework_proxy_rewards_progress_and_penalizes_oversized_edits(self):
        current = "a\nb\n"
        target = "a\nx\nb\n"
        task = first_task(["a\n", current, target, target])

        no_change = score_prediction(
            task, current, cohort="test", model_name="no_change"
        )
        exact = score_prediction(
            task, target, cohort="test", model_name="exact"
        )
        oversized = score_prediction(
            task, "extra\na\nx\nb\nmore\n", cohort="test", model_name="oversized"
        )

        self.assertEqual(line_edit_cost(current, target), 1)
        self.assertEqual(no_change["rework_reduction"], 0.0)
        self.assertEqual(exact["rework_reduction"], 1.0)
        self.assertLess(oversized["rework_reduction"], 0.0)
        self.assertTrue(oversized["rework_worsened"])

    def test_stable_reference_requires_observed_follow_up(self):
        stable = first_task(
            ["old\n", "base\n", "base\nx\n", "base\nx\n"]
        )
        stable_score = score_prediction(
            stable, stable["target"], cohort="test", model_name="model"
        )
        self.assertTrue(stable_score["persistence_eligible"])
        self.assertEqual(stable_score["stable_reference_blocks"], 1)
        self.assertEqual(stable_score["stable_precision"], 1.0)
        self.assertEqual(stable_score["stable_recall"], 1.0)

        reverted = first_task(
            ["old\n", "base\n", "base\nx\n", "base\n"]
        )
        reverted_score = score_prediction(
            reverted, reverted["target"], cohort="test", model_name="model"
        )
        self.assertTrue(reverted_score["persistence_eligible"])
        self.assertEqual(reverted_score["stable_reference_blocks"], 0)
        self.assertEqual(reverted_score["stable_precision"], 0.0)
        self.assertTrue(math.isnan(reverted_score["stable_recall"]))
        self.assertEqual(reverted_score["stable_f1"], 0.0)

        censored = first_task(["old\n", "base\n", "base\nx\n"])
        censored_score = score_prediction(
            censored, censored["target"], cohort="test", model_name="model"
        )
        self.assertTrue(censored_score["persistence_censored"])
        self.assertEqual(censored_score["persistence_censored_blocks"], 1)

    def test_persistence_threshold_changes_eligibility(self):
        sessions = [
            {"session_id": "s1", "states": ["a", "b", "c", "d"]}
        ]
        result = persistence_threshold_analysis(
            sessions,
            {"copy_target": lambda task: task["target"]},
            cohort="test",
            thresholds=(2, 3),
        )
        by_threshold = result.set_index("persistence_min_consecutive")
        self.assertEqual(by_threshold.loc[2, "persistence_eligible_tasks"], 1)
        self.assertEqual(by_threshold.loc[3, "persistence_eligible_tasks"], 0)

    def test_cluster_bootstrap_keeps_models_paired_by_trajectory(self):
        sessions = [
            {"session_id": "s1", "states": ["a", "b", "c", "d"]},
            {"session_id": "s2", "states": ["x", "y", "z", "w"]},
        ]
        detailed = evaluate_predictors_detailed(
            sessions,
            (1,),
            "test",
            {
                "no_change": lambda task: task["current"],
                "copy_target": lambda task: task["target"],
            },
        )
        comparison = paired_cluster_bootstrap(
            detailed,
            "no_change",
            "copy_target",
            cohort="test",
            n_bootstrap=100,
            seed=7,
        ).set_index("metric")
        self.assertEqual(
            comparison.loc["exact_match", "n_trajectory_clusters"], 2
        )
        self.assertEqual(comparison.loc["exact_match", "n_paired_tasks"], 4)
        self.assertEqual(comparison.loc["exact_match", "delta_b_minus_a"], 1.0)
        self.assertEqual(comparison.loc["exact_match", "ci_95_low"], 1.0)
        self.assertEqual(comparison.loc["exact_match", "ci_95_high"], 1.0)
        self.assertEqual(
            comparison.loc["pooled_rework_reduction", "delta_b_minus_a"], 1.0
        )

    def test_exact_one_step_reversion_and_censoring_are_separate(self):
        reverted = first_task(["old\n", "base\n", "base\nx\n", "base\n"])
        reverted_score = score_prediction(
            reverted, reverted["target"], cohort="test", model_name="model"
        )
        self.assertTrue(reverted_score["reversion_eligible"])
        self.assertTrue(reverted_score["exact_one_step_reversion"])
        self.assertFalse(reverted_score["reversion_censored"])

        censored = first_task(["old\n", "base\n", "base\nx\n"])
        censored_score = score_prediction(
            censored, censored["target"], cohort="test", model_name="model"
        )
        self.assertFalse(censored_score["reversion_eligible"])
        self.assertTrue(censored_score["reversion_censored"])

    def test_invalid_external_prediction_reduces_coverage_not_accuracy(self):
        sessions = [{"session_id": "s1", "states": ["a", "b", "c"]}]
        predictions = pd.DataFrame(
            [
                {
                    "example_id": "s1::a1::h1",
                    "model_name": "model",
                    "horizon": 1,
                    "prediction": None,
                    "generation_status": "error:timeout",
                }
            ]
        )
        detailed, summary = evaluate_external_predictions(
            sessions, predictions, cohort="test"
        )
        self.assertFalse(detailed.iloc[0]["scored"])
        self.assertEqual(summary.iloc[0]["invalid_predictions"], 1)
        self.assertEqual(summary.iloc[0]["prediction_coverage"], 0.0)
        self.assertTrue(math.isnan(summary.iloc[0]["exact_match"]))

    def test_missing_prediction_row_is_counted_in_coverage(self):
        sessions = [{"session_id": "s1", "states": ["a", "b", "c", "d"]}]
        predictions = pd.DataFrame(
            [
                {
                    "example_id": "s1::a1::h1",
                    "model_name": "model",
                    "horizon": 1,
                    "prediction": "c",
                    "generation_status": "ok",
                }
            ]
        )
        detailed, summary = evaluate_external_predictions(
            sessions, predictions, cohort="test"
        )
        self.assertEqual(len(detailed), 2)
        self.assertEqual((detailed["generation_status"] == "missing").sum(), 1)
        self.assertEqual(summary.iloc[0]["prediction_coverage"], 0.5)

    def test_external_contract_rejects_duplicate_predictions(self):
        sessions = [{"session_id": "s1", "states": ["a", "b", "c"]}]
        row = {
            "example_id": "s1::a1::h1",
            "model_name": "model",
            "horizon": 1,
            "prediction": "c",
            "generation_status": "ok",
        }
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_external_predictions(
                sessions, pd.DataFrame([row, row]), cohort="test"
            )

    def test_external_contract_rejects_unknown_example(self):
        sessions = [{"session_id": "s1", "states": ["a", "b", "c"]}]
        predictions = pd.DataFrame(
            [
                {
                    "example_id": "missing::a1::h1",
                    "model_name": "model",
                    "horizon": 1,
                    "prediction": "c",
                    "generation_status": "ok",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "Unknown example_id"):
            evaluate_external_predictions(sessions, predictions, cohort="test")


if __name__ == "__main__":
    unittest.main()
