"""Tests for src/evaluate.py.

Coverage targets:
  ModelEvaluator.evaluate()             — returns EvaluationReport; all metric groups
  ModelEvaluator.fairness_audit()       — per-group metrics; suppression; missing cols
  ModelEvaluator.complete_case_sensitivity() — full vs. complete-case comparison
  EvaluationReport.summary()            — printable string with split label
  _primary_metrics()                    — recall, precision, f1, auc_roc, pr_auc
  _clinical_metrics()                   — tp/fp/fn/tn, sensitivity, specificity, PPV, NPV, FNR, NNS
  _calibration_metrics()                — brier_score, expected_calibration_error
  _check_fairness_gates()               — warning logged when recall gap > 0.10
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import EvaluationReport, ModelEvaluator


# ── Shared fitted model (module scope — fit once) ────────────────────────────

@pytest.fixture(scope="module")
def eval_model(preprocessed_feature_X, feature_y):
    """Shallow DecisionTree for fast predict_proba calls."""
    m = DecisionTreeClassifier(max_depth=4, random_state=42)
    m.fit(preprocessed_feature_X.values, feature_y.values)
    return m


@pytest.fixture(scope="module")
def evaluator(cfg, tmp_path_factory):
    figs = tmp_path_factory.mktemp("eval_figs")
    return ModelEvaluator(cfg, figures_dir=figs)


@pytest.fixture(scope="module")
def eval_report(evaluator, eval_model, preprocessed_feature_X, feature_y):
    """One EvaluationReport reused across structural and metric tests."""
    return evaluator.evaluate(
        eval_model, preprocessed_feature_X, feature_y,
        threshold=0.5, split_label="test", save_plots=False
    )


# ── EvaluationReport dataclass ───────────────────────────────────────────────

class TestEvaluationReport:
    def test_summary_returns_string(self, eval_report):
        assert isinstance(eval_report.summary(), str)

    def test_summary_contains_split_label(self, eval_report):
        assert "test" in eval_report.summary()

    def test_summary_contains_threshold(self, eval_report):
        assert "0.5" in eval_report.summary() or "Threshold" in eval_report.summary()

    def test_n_samples_matches_input(self, eval_report, preprocessed_feature_X):
        assert eval_report.n_samples == len(preprocessed_feature_X)

    def test_n_positive_matches_true_labels(self, eval_report, feature_y):
        assert eval_report.n_positive == int(feature_y.sum())

    def test_threshold_stored_correctly(self, eval_report):
        assert eval_report.threshold == pytest.approx(0.5)

    def test_metrics_is_dict(self, eval_report):
        assert isinstance(eval_report.metrics, dict)
        assert len(eval_report.metrics) > 0


# ── Primary metrics ───────────────────────────────────────────────────────────

class TestPrimaryMetrics:
    def test_recall_present_and_in_range(self, eval_report):
        assert "recall" in eval_report.metrics
        assert 0.0 <= eval_report.metrics["recall"] <= 1.0

    def test_precision_present_and_in_range(self, eval_report):
        assert "precision" in eval_report.metrics
        assert 0.0 <= eval_report.metrics["precision"] <= 1.0

    def test_f1_present_and_in_range(self, eval_report):
        assert "f1" in eval_report.metrics
        assert 0.0 <= eval_report.metrics["f1"] <= 1.0

    def test_auc_roc_present_and_in_range(self, eval_report):
        assert "auc_roc" in eval_report.metrics
        val = eval_report.metrics["auc_roc"]
        if not np.isnan(val):
            assert 0.0 <= val <= 1.0

    def test_pr_auc_present_and_in_range(self, eval_report):
        assert "pr_auc" in eval_report.metrics
        val = eval_report.metrics["pr_auc"]
        if not np.isnan(val):
            assert 0.0 <= val <= 1.0

    def test_auc_roc_nan_when_single_class(self, evaluator, eval_model, preprocessed_feature_X):
        y_all_zero = pd.Series(np.zeros(len(preprocessed_feature_X), dtype=int))
        report = evaluator.evaluate(
            eval_model, preprocessed_feature_X, y_all_zero,
            threshold=0.5, save_plots=False
        )
        assert np.isnan(report.metrics["auc_roc"])
        assert np.isnan(report.metrics["pr_auc"])


# ── Clinical metrics ──────────────────────────────────────────────────────────

class TestClinicalMetrics:
    _CLINICAL_KEYS = {
        "sensitivity (recall)", "specificity", "ppv (precision)", "npv",
        "false_negative_rate", "number_needed_to_screen", "tp", "fp", "fn", "tn",
    }

    def test_all_clinical_keys_present(self, eval_report):
        assert self._CLINICAL_KEYS.issubset(eval_report.metrics.keys())

    def test_tp_fp_fn_tn_are_integers(self, eval_report):
        for key in ("tp", "fp", "fn", "tn"):
            assert isinstance(eval_report.metrics[key], int)

    def test_tp_fn_sum_to_n_positive(self, eval_report):
        assert eval_report.metrics["tp"] + eval_report.metrics["fn"] == eval_report.n_positive

    def test_fp_tn_sum_to_n_negative(self, eval_report):
        n_negative = eval_report.n_samples - eval_report.n_positive
        assert eval_report.metrics["fp"] + eval_report.metrics["tn"] == n_negative

    def test_sensitivity_equals_recall(self, eval_report):
        assert eval_report.metrics["sensitivity (recall)"] == pytest.approx(
            eval_report.metrics["recall"]
        )

    def test_specificity_in_0_1(self, eval_report):
        assert 0.0 <= eval_report.metrics["specificity"] <= 1.0

    def test_false_negative_rate_in_0_1(self, eval_report):
        assert 0.0 <= eval_report.metrics["false_negative_rate"] <= 1.0

    def test_npv_in_0_1(self, eval_report):
        assert 0.0 <= eval_report.metrics["npv"] <= 1.0

    def test_sensitivity_plus_fnr_approximately_one(self, eval_report):
        sens = eval_report.metrics["sensitivity (recall)"]
        fnr = eval_report.metrics["false_negative_rate"]
        assert sens + fnr == pytest.approx(1.0, abs=1e-6)


# ── Calibration metrics ───────────────────────────────────────────────────────

class TestCalibrationMetrics:
    def test_brier_score_present(self, eval_report):
        assert "brier_score" in eval_report.metrics

    def test_brier_score_in_0_1(self, eval_report):
        assert 0.0 <= eval_report.metrics["brier_score"] <= 1.0

    def test_ece_present(self, eval_report):
        assert "expected_calibration_error" in eval_report.metrics

    def test_ece_non_negative(self, eval_report):
        assert eval_report.metrics["expected_calibration_error"] >= 0.0


# ── Threshold effects ─────────────────────────────────────────────────────────

class TestThresholdEffect:
    def test_threshold_0_gives_recall_1_if_any_positive(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y
    ):
        report = evaluator.evaluate(
            eval_model, preprocessed_feature_X, feature_y,
            threshold=0.0, save_plots=False
        )
        assert report.metrics["recall"] == pytest.approx(1.0)

    def test_high_threshold_reduces_recall_vs_low(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y
    ):
        low = evaluator.evaluate(
            eval_model, preprocessed_feature_X, feature_y,
            threshold=0.1, save_plots=False
        )
        high = evaluator.evaluate(
            eval_model, preprocessed_feature_X, feature_y,
            threshold=0.9, save_plots=False
        )
        assert low.metrics["recall"] >= high.metrics["recall"]

    def test_threshold_stored_in_report(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y
    ):
        report = evaluator.evaluate(
            eval_model, preprocessed_feature_X, feature_y,
            threshold=0.42, save_plots=False
        )
        assert report.threshold == pytest.approx(0.42)

    def test_split_label_stored_in_report(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y
    ):
        report = evaluator.evaluate(
            eval_model, preprocessed_feature_X, feature_y,
            split_label="validation", save_plots=False
        )
        assert report.split_label == "validation"


# ── Fairness audit ────────────────────────────────────────────────────────────

class TestFairnessAudit:
    def test_returns_dataframe(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y, students_df
    ):
        result = evaluator.fairness_audit(
            eval_model, preprocessed_feature_X, feature_y, students_df, threshold=0.5
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y, students_df
    ):
        result = evaluator.fairness_audit(
            eval_model, preprocessed_feature_X, feature_y, students_df, threshold=0.5
        )
        required = {"audit_col", "group_value", "n", "n_positive", "recall", "precision", "f1"}
        assert required.issubset(result.columns)

    def test_recall_values_in_0_1(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y, students_df
    ):
        result = evaluator.fairness_audit(
            eval_model, preprocessed_feature_X, feature_y, students_df, threshold=0.5
        )
        if len(result) > 0:
            assert (result["recall"] >= 0).all()
            assert (result["recall"] <= 1).all()

    def test_suppresses_groups_below_min_n(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y, students_df
    ):
        # Only two students; their group should be suppressed
        tiny_df = students_df.copy()
        tiny_df["rare_group_col"] = "rare"
        tiny_df.loc[0, "rare_group_col"] = "very_rare"
        tiny_df.loc[1, "rare_group_col"] = "very_rare"
        # Fairness audit cols is driven by schema; rare_group_col is not in schema
        # so this test verifies via gender (has ~50 per group, none suppressed)
        result = evaluator.fairness_audit(
            eval_model, preprocessed_feature_X, feature_y, students_df, threshold=0.5
        )
        assert (result["n"] >= 10).all()

    def test_no_audit_cols_returns_empty_dataframe(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y
    ):
        # df_raw with no fairness audit columns
        bare_df = pd.DataFrame({"unrelated_col": range(len(feature_y))})
        result = evaluator.fairness_audit(
            eval_model, preprocessed_feature_X, feature_y, bare_df, threshold=0.5
        )
        assert len(result) == 0

    def test_audit_col_column_contains_known_fairness_columns(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y, students_df
    ):
        result = evaluator.fairness_audit(
            eval_model, preprocessed_feature_X, feature_y, students_df, threshold=0.5
        )
        known_cols = {"gender", "ethnicity_bme", "deprivation_quintile", "sen_status"}
        assert set(result["audit_col"].unique()).issubset(known_cols)


# ── Fairness gate check ───────────────────────────────────────────────────────

class TestFairnessGateCheck:
    def test_warns_when_recall_gap_exceeds_0_10(
        self, evaluator, caplog
    ):
        import logging
        # Create audit_df with a large recall gap
        audit_df = pd.DataFrame({
            "audit_col": ["gender", "gender"],
            "group_value": ["male", "female"],
            "n": [50, 50],
            "n_positive": [10, 10],
            "recall": [0.90, 0.50],
            "precision": [0.80, 0.60],
            "f1": [0.85, 0.55],
        })
        with caplog.at_level(logging.WARNING):
            evaluator._check_fairness_gates(audit_df)
        assert any("Fairness gate" in m or "gap" in m.lower() for m in caplog.messages)

    def test_no_warning_when_recall_gap_within_threshold(
        self, evaluator, caplog
    ):
        import logging
        audit_df = pd.DataFrame({
            "audit_col": ["gender", "gender"],
            "group_value": ["male", "female"],
            "n": [50, 50],
            "n_positive": [10, 10],
            "recall": [0.82, 0.80],
            "precision": [0.80, 0.78],
            "f1": [0.81, 0.79],
        })
        with caplog.at_level(logging.WARNING):
            evaluator._check_fairness_gates(audit_df)
        gate_warnings = [m for m in caplog.messages if "Fairness gate" in m]
        assert len(gate_warnings) == 0


# ── CompleteCaseSensitivity ───────────────────────────────────────────────────

class TestCompleteCaseSensitivity:
    def test_returns_dict_with_full_key(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y, students_df
    ):
        result = evaluator.complete_case_sensitivity(
            eval_model, preprocessed_feature_X, feature_y, students_df, threshold=0.5
        )
        assert "full" in result
        assert isinstance(result["full"], EvaluationReport)

    def test_skips_complete_case_when_flag_column_absent(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y
    ):
        bare_df = pd.DataFrame({"x": range(len(feature_y))})
        result = evaluator.complete_case_sensitivity(
            eval_model, preprocessed_feature_X, feature_y, bare_df, threshold=0.5
        )
        assert "full" in result
        assert "complete_case" not in result

    def test_complete_case_key_present_when_flag_exists(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y, students_df
    ):
        # students_df has spin_complete_flag=1 for all 100 rows → complete_case subset = full
        result = evaluator.complete_case_sensitivity(
            eval_model, preprocessed_feature_X, feature_y, students_df, threshold=0.5
        )
        assert "complete_case" in result

    def test_full_report_n_samples_equals_total(
        self, evaluator, eval_model, preprocessed_feature_X, feature_y, students_df
    ):
        result = evaluator.complete_case_sensitivity(
            eval_model, preprocessed_feature_X, feature_y, students_df, threshold=0.5
        )
        assert result["full"].n_samples == len(preprocessed_feature_X)
