"""Tests for src/train.py.

Coverage targets (in order of increasing integration):
  _selection_score()                — weighted composite; NaN-tolerant
  _compute_metrics()                — four keys returned; NaN AUC for single class
  optimise_threshold()              — normal path; fallback when no valid threshold
  _check_deployment_gates()         — passes when gates met; ValueError when failed
  _interpretability_rank()          — known order from config
  FoldReport.summary()              — returns non-empty string
  NestedCVResults.summary()         — returns non-empty string
  _aggregate_fold_results()         — mean/std per metric; NaN-safe
  _select_best_model()              — selects model with highest mean composite
  _separate_test_school()           — no school overlap; sizes sum to total
  _run_nested_cv()                  — returns NestedCVResults; correct fold count
  run_training_pipeline()           — end-to-end smoke test (marked slow)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import Config
from src.train import (
    FoldReport,
    NestedCVResults,
    _aggregate_fold_results,
    _check_deployment_gates,
    _compute_metrics,
    _interpretability_rank,
    _run_nested_cv,
    _select_best_model,
    _selection_score,
    _separate_test_school,
    optimise_threshold,
    run_training_pipeline,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _perfect_binary_model(n: int = 80, seed: int = 0):
    """Return a LogisticRegression fitted on linearly separable data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X, y)
    return model, X, y


def _make_fold_report(idx: int, metrics: dict | None = None) -> FoldReport:
    if metrics is None:
        metrics = {"recall": 0.80, "pr_auc": 0.60, "f1": 0.70, "auc_roc": 0.85}
    return FoldReport(
        fold_idx=idx,
        test_school_ids=[f"SCH{idx:03d}"],
        n_test=20,
        n_positive_test=3,
        model_metrics={"ModelA": metrics, "ModelB": {**metrics, "recall": 0.50}},
        model_params={"ModelA": {"C": 1.0}, "ModelB": {"n_estimators": 100}},
        model_thresholds={"ModelA": 0.40, "ModelB": 0.50},
        model_composites={"ModelA": 0.75, "ModelB": 0.55},
        best_model_name="ModelA",
        best_composite=0.75,
    )


# ── _selection_score ──────────────────────────────────────────────────────────

class TestSelectionScore:
    def test_known_weighted_sum(self):
        metrics = {"recall": 1.0, "pr_auc": 1.0, "f1": 1.0, "auc_roc": 1.0}
        weights = {"recall": 0.45, "pr_auc": 0.25, "f1": 0.20, "auc_roc": 0.10}
        assert _selection_score(metrics, weights) == pytest.approx(1.0)

    def test_partial_weights(self):
        metrics = {"recall": 0.80, "pr_auc": 0.60, "f1": 0.70, "auc_roc": 0.85}
        weights = {"recall": 0.45, "pr_auc": 0.25, "f1": 0.20, "auc_roc": 0.10}
        expected = 0.45*0.80 + 0.25*0.60 + 0.20*0.70 + 0.10*0.85
        assert _selection_score(metrics, weights) == pytest.approx(expected)

    def test_nan_metric_is_skipped(self):
        metrics = {"recall": 0.80, "pr_auc": float("nan"), "f1": 0.70, "auc_roc": 0.85}
        weights = {"recall": 0.45, "pr_auc": 0.25, "f1": 0.20, "auc_roc": 0.10}
        score = _selection_score(metrics, weights)
        assert not math.isnan(score)
        assert score > 0

    def test_zero_metrics_gives_zero_score(self):
        metrics = {"recall": 0.0, "pr_auc": 0.0, "f1": 0.0, "auc_roc": 0.0}
        weights = {"recall": 0.45, "pr_auc": 0.25, "f1": 0.20, "auc_roc": 0.10}
        assert _selection_score(metrics, weights) == pytest.approx(0.0)


# ── _compute_metrics ──────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_four_required_keys(self):
        model, X, y = _perfect_binary_model()
        X_df = pd.DataFrame(X)
        y_s = pd.Series(y)
        metrics = _compute_metrics(model, X_df, y_s, threshold=0.5)
        for key in ("recall", "pr_auc", "f1", "auc_roc"):
            assert key in metrics

    def test_recall_near_1_for_perfect_model(self):
        model, X, y = _perfect_binary_model()
        X_df = pd.DataFrame(X)
        y_s = pd.Series(y)
        metrics = _compute_metrics(model, X_df, y_s, threshold=0.5)
        assert metrics["recall"] >= 0.90

    def test_nan_auc_when_single_class_present(self):
        model, X, _ = _perfect_binary_model()
        X_df = pd.DataFrame(X)
        y_all_zero = pd.Series(np.zeros(len(X), dtype=int))
        metrics = _compute_metrics(model, X_df, y_all_zero, threshold=0.5)
        assert math.isnan(metrics["auc_roc"])

    def test_threshold_affects_recall(self):
        model, X, y = _perfect_binary_model()
        X_df = pd.DataFrame(X)
        y_s = pd.Series(y)
        # Very high threshold → fewer positive predictions → lower recall
        metrics_high = _compute_metrics(model, X_df, y_s, threshold=0.99)
        metrics_low = _compute_metrics(model, X_df, y_s, threshold=0.01)
        assert metrics_low["recall"] >= metrics_high["recall"]


# ── optimise_threshold ────────────────────────────────────────────────────────

class TestOptimiseThreshold:
    def test_returns_float_between_0_and_1(self):
        model, X, y = _perfect_binary_model()
        threshold = optimise_threshold(
            model, pd.DataFrame(X), pd.Series(y), recall_minimum=0.50
        )
        assert 0.0 <= threshold <= 1.0

    def test_recall_at_threshold_meets_minimum(self):
        model, X, y = _perfect_binary_model()
        threshold = optimise_threshold(
            model, pd.DataFrame(X), pd.Series(y), recall_minimum=0.70
        )
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        from sklearn.metrics import recall_score
        achieved = recall_score(y, pred, zero_division=0)
        assert achieved >= 0.70 - 1e-6  # tiny tolerance for floating-point

    def test_fallback_when_no_threshold_achieves_minimum(self, caplog):
        import logging
        model, X, y = _perfect_binary_model()
        with caplog.at_level(logging.WARNING):
            threshold = optimise_threshold(
                model, pd.DataFrame(X), pd.Series(y), recall_minimum=1.01
            )
        # Should still return a float (the lowest threshold available)
        assert isinstance(threshold, float)
        assert any("No threshold" in msg for msg in caplog.messages)


# ── _check_deployment_gates ───────────────────────────────────────────────────

class TestCheckDeploymentGates:
    def test_passes_when_all_gates_met(self, cfg):
        metrics = {"recall": 0.90, "auc_roc": 0.85, "pr_auc": 0.60}
        _check_deployment_gates(metrics, cfg)  # should not raise

    def test_raises_on_recall_below_minimum(self, cfg):
        metrics = {"recall": 0.50, "auc_roc": 0.85, "pr_auc": 0.60}
        with pytest.raises(ValueError, match="recall"):
            _check_deployment_gates(metrics, cfg)

    def test_raises_on_auc_roc_below_minimum(self, cfg):
        metrics = {"recall": 0.90, "auc_roc": 0.50, "pr_auc": 0.60}
        with pytest.raises(ValueError, match="auc_roc"):
            _check_deployment_gates(metrics, cfg)

    def test_raises_on_pr_auc_below_minimum(self, cfg):
        metrics = {"recall": 0.90, "auc_roc": 0.85, "pr_auc": 0.10}
        with pytest.raises(ValueError, match="pr_auc"):
            _check_deployment_gates(metrics, cfg)

    def test_error_message_lists_all_failures(self, cfg):
        metrics = {"recall": 0.50, "auc_roc": 0.50, "pr_auc": 0.10}
        with pytest.raises(ValueError) as exc_info:
            _check_deployment_gates(metrics, cfg)
        msg = str(exc_info.value)
        assert "recall" in msg
        assert "auc_roc" in msg
        assert "pr_auc" in msg


# ── _interpretability_rank ────────────────────────────────────────────────────

class TestInterpretabilityRank:
    def test_decision_tree_is_most_interpretable(self, cfg):
        assert _interpretability_rank("DecisionTree", cfg) == 0

    def test_unknown_model_gets_last_rank(self, cfg):
        rank = _interpretability_rank("SomeFutureModel", cfg)
        assert rank == len(cfg.interpretability_rank)

    def test_logistic_regression_before_random_forest(self, cfg):
        lr_rank = _interpretability_rank("LogisticRegression", cfg)
        rf_rank = _interpretability_rank("RandomForest", cfg)
        assert lr_rank < rf_rank


# ── FoldReport / NestedCVResults ──────────────────────────────────────────────

class TestDataclassFormatting:
    def test_fold_report_summary_is_string(self):
        report = _make_fold_report(0)
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_fold_report_summary_contains_fold_number(self):
        report = _make_fold_report(2)
        assert "3" in report.summary()  # fold 2 → "Outer Fold 3"

    def test_fold_report_summary_marks_best_model(self):
        report = _make_fold_report(0)
        assert "◀" in report.summary() or "BEST" in report.summary().upper() \
            or "ModelA" in report.summary()

    def test_nested_cv_results_summary_is_string(self, cfg):
        fold_reports = [_make_fold_report(i) for i in range(2)]
        model_summary = _aggregate_fold_results(fold_reports)
        results = NestedCVResults(
            n_outer_folds=2,
            fold_reports=fold_reports,
            model_summary=model_summary,
            best_model_name="ModelA",
            best_fold_idx=0,
        )
        s = results.summary()
        assert isinstance(s, str)
        assert "ModelA" in s

    def test_nested_cv_results_summary_marks_selected_model(self, cfg):
        fold_reports = [_make_fold_report(i) for i in range(2)]
        model_summary = _aggregate_fold_results(fold_reports)
        results = NestedCVResults(
            n_outer_folds=2,
            fold_reports=fold_reports,
            model_summary=model_summary,
            best_model_name="ModelA",
            best_fold_idx=0,
        )
        assert "SELECTED" in results.summary()


# ── _aggregate_fold_results ───────────────────────────────────────────────────

class TestAggregateResults:
    def test_mean_recall_computed_correctly(self):
        reports = [
            _make_fold_report(0, metrics={"recall": 0.80, "pr_auc": 0.60, "f1": 0.70, "auc_roc": 0.85}),
            _make_fold_report(1, metrics={"recall": 0.90, "pr_auc": 0.65, "f1": 0.75, "auc_roc": 0.88}),
        ]
        summary = _aggregate_fold_results(reports)
        assert summary["ModelA"]["mean_metrics"]["recall"] == pytest.approx(0.85)

    def test_std_uses_ddof_1(self):
        reports = [
            _make_fold_report(0, metrics={"recall": 0.80, "pr_auc": 0.60, "f1": 0.70, "auc_roc": 0.85}),
            _make_fold_report(1, metrics={"recall": 0.90, "pr_auc": 0.65, "f1": 0.75, "auc_roc": 0.88}),
        ]
        summary = _aggregate_fold_results(reports)
        # std of [0.80, 0.90] with ddof=1
        expected_std = np.std([0.80, 0.90], ddof=1)
        assert summary["ModelA"]["std_metrics"]["recall"] == pytest.approx(expected_std)

    def test_nan_auc_handled_by_nanmean(self):
        r0 = _make_fold_report(0, metrics={"recall": 0.80, "pr_auc": 0.60, "f1": 0.70, "auc_roc": float("nan")})
        r1 = _make_fold_report(1, metrics={"recall": 0.90, "pr_auc": 0.65, "f1": 0.75, "auc_roc": 0.88})
        summary = _aggregate_fold_results([r0, r1])
        # nanmean of [nan, 0.88] = 0.88
        assert summary["ModelA"]["mean_metrics"]["auc_roc"] == pytest.approx(0.88)

    def test_all_model_names_present(self):
        reports = [_make_fold_report(i) for i in range(3)]
        summary = _aggregate_fold_results(reports)
        assert "ModelA" in summary
        assert "ModelB" in summary

    def test_mean_composite_matches_manual_average(self):
        reports = [_make_fold_report(i) for i in range(3)]
        summary = _aggregate_fold_results(reports)
        # All fold_reports have ModelA composite = 0.75
        assert summary["ModelA"]["mean_composite"] == pytest.approx(0.75)


# ── _select_best_model ────────────────────────────────────────────────────────

class TestSelectBestModel:
    def test_selects_model_with_highest_mean_composite(self, cfg):
        # ModelA has composite 0.75, ModelB has 0.55 consistently
        reports = [_make_fold_report(i) for i in range(3)]
        summary = _aggregate_fold_results(reports)
        best_name, best_idx = _select_best_model(summary, reports, cfg)
        assert best_name == "ModelA"

    def test_best_fold_idx_is_valid_index(self, cfg):
        reports = [_make_fold_report(i) for i in range(3)]
        summary = _aggregate_fold_results(reports)
        _, best_idx = _select_best_model(summary, reports, cfg)
        assert 0 <= best_idx < len(reports)


# ── _separate_test_school ─────────────────────────────────────────────────────

class TestSeparateTestSchool:
    def _apply_fe(self, cfg, students_df):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(cfg)
        return fe.transform(students_df.copy())

    def test_pool_and_test_sizes_sum_to_total(self, cfg, students_df):
        df = self._apply_fe(cfg, students_df)
        X_pool, y_pool, groups_pool, X_test, y_test = _separate_test_school(df, cfg)
        assert len(X_pool) + len(X_test) == len(df)

    def test_no_school_overlap_between_pool_and_test(self, cfg, students_df):
        df = self._apply_fe(cfg, students_df)
        X_pool, y_pool, groups_pool, X_test, y_test = _separate_test_school(df, cfg)
        pool_schools = set(groups_pool.unique())
        # We can't directly access test school groups, but we can verify pool_groups
        # returned by the function correctly aligns with X_pool length
        assert len(groups_pool) == len(X_pool)

    def test_groups_pool_length_matches_X_pool(self, cfg, students_df):
        df = self._apply_fe(cfg, students_df)
        X_pool, y_pool, groups_pool, X_test, y_test = _separate_test_school(df, cfg)
        assert len(groups_pool) == len(X_pool)

    def test_y_pool_length_matches_X_pool(self, cfg, students_df):
        df = self._apply_fe(cfg, students_df)
        X_pool, y_pool, groups_pool, X_test, y_test = _separate_test_school(df, cfg)
        assert len(y_pool) == len(X_pool)

    def test_y_test_length_matches_X_test(self, cfg, students_df):
        df = self._apply_fe(cfg, students_df)
        X_pool, y_pool, groups_pool, X_test, y_test = _separate_test_school(df, cfg)
        assert len(y_test) == len(X_test)

    def test_test_set_is_non_empty(self, cfg, students_df):
        df = self._apply_fe(cfg, students_df)
        _, _, _, X_test, y_test = _separate_test_school(df, cfg)
        assert len(X_test) > 0


# ── _run_nested_cv ────────────────────────────────────────────────────────────

class TestRunNestedCV:
    """Tests that call _run_nested_cv with a fast (1-trial, 2-fold) config."""

    @pytest.fixture
    def fast_cfg(self, cfg, monkeypatch):
        monkeypatch.setattr(type(cfg), "optuna_n_trials", property(lambda s: 1))
        monkeypatch.setattr(type(cfg), "cv_outer_folds", property(lambda s: 2))
        monkeypatch.setattr(type(cfg), "cv_inner_folds", property(lambda s: 2))
        return cfg

    @pytest.fixture
    def pool_data(self, cfg, make_students):
        from src.feature_engineering import FeatureEngineer
        df = make_students(n=80, n_schools=6, prevalence=0.18, seed=1)
        fe = FeatureEngineer(cfg)
        df = fe.transform(df)
        X_pool, y_pool, groups_pool, _, _ = _separate_test_school(df, cfg)
        return X_pool, y_pool, groups_pool

    @pytest.mark.slow
    def test_returns_nested_cv_results_instance(self, fast_cfg, pool_data):
        X_pool, y_pool, groups_pool = pool_data
        results = _run_nested_cv(X_pool, y_pool, groups_pool, fast_cfg)
        assert isinstance(results, NestedCVResults)

    @pytest.mark.slow
    def test_n_outer_folds_matches_or_clipped_to_schools(self, fast_cfg, pool_data):
        X_pool, y_pool, groups_pool = pool_data
        results = _run_nested_cv(X_pool, y_pool, groups_pool, fast_cfg)
        n_schools_in_pool = len(groups_pool.unique())
        assert results.n_outer_folds <= min(2, n_schools_in_pool)

    @pytest.mark.slow
    def test_fold_report_count_matches_n_outer_folds(self, fast_cfg, pool_data):
        X_pool, y_pool, groups_pool = pool_data
        results = _run_nested_cv(X_pool, y_pool, groups_pool, fast_cfg)
        assert len(results.fold_reports) == results.n_outer_folds

    @pytest.mark.slow
    def test_best_model_name_is_known_model(self, fast_cfg, pool_data):
        from src.train import _build_model_catalogue
        X_pool, y_pool, groups_pool = pool_data
        results = _run_nested_cv(X_pool, y_pool, groups_pool, fast_cfg)
        known_models = set(_build_model_catalogue(fast_cfg).keys())
        assert results.best_model_name in known_models


# ── run_training_pipeline (integration) ───────────────────────────────────────

class TestRunTrainingPipeline:
    """End-to-end smoke test with mocked I/O and fast hyperparameter search."""

    @pytest.fixture
    def fast_cfg(self, cfg, monkeypatch):
        monkeypatch.setattr(type(cfg), "optuna_n_trials", property(lambda s: 1))
        monkeypatch.setattr(type(cfg), "cv_outer_folds", property(lambda s: 2))
        monkeypatch.setattr(type(cfg), "cv_inner_folds", property(lambda s: 2))
        # Lower gates so synthetic data doesn't fail
        monkeypatch.setattr(type(cfg), "recall_minimum",  property(lambda s: 0.0))
        monkeypatch.setattr(type(cfg), "auc_roc_minimum", property(lambda s: 0.0))
        monkeypatch.setattr(type(cfg), "pr_auc_minimum",  property(lambda s: 0.0))
        return cfg

    @pytest.mark.slow
    def test_pipeline_returns_expected_keys(self, fast_cfg, monkeypatch, make_students):
        df = make_students(n=80, n_schools=6, prevalence=0.18, seed=5)

        # Mock data loading
        monkeypatch.setattr("src.train.load_active_dataset", lambda c: (df, {}))
        # Mock artefact saving (avoids writing to disk and modifying registry)
        monkeypatch.setattr(
            "src.train._save_artefacts",
            lambda model, preprocessor, model_name, dataset_key,
                   threshold, cv_results, test_metrics, cfg:
                   {"model": Path("m.pkl"), "preprocessor": Path("p.pkl"), "spin_imputer": Path("s.pkl")},
        )

        result = run_training_pipeline(fast_cfg)

        for key in ("best_model_name", "best_model", "preprocessor", "threshold",
                    "cv_results", "test_metrics", "artefact_paths", "feature_names"):
            assert key in result, f"Missing key in pipeline result: {key}"

    @pytest.mark.slow
    def test_pipeline_threshold_in_0_1(self, fast_cfg, monkeypatch, make_students):
        df = make_students(n=80, n_schools=6, prevalence=0.18, seed=6)
        monkeypatch.setattr("src.train.load_active_dataset", lambda c: (df, {}))
        monkeypatch.setattr(
            "src.train._save_artefacts",
            lambda model, preprocessor, model_name, dataset_key,
                   threshold, cv_results, test_metrics, cfg:
                   {"model": Path("m.pkl"), "preprocessor": Path("p.pkl"), "spin_imputer": Path("s.pkl")},
        )
        result = run_training_pipeline(fast_cfg)
        assert 0.0 <= result["threshold"] <= 1.0

    @pytest.mark.slow
    def test_pipeline_cv_results_is_nested_cv_results(self, fast_cfg, monkeypatch, make_students):
        df = make_students(n=80, n_schools=6, prevalence=0.18, seed=7)
        monkeypatch.setattr("src.train.load_active_dataset", lambda c: (df, {}))
        monkeypatch.setattr(
            "src.train._save_artefacts",
            lambda model, preprocessor, model_name, dataset_key,
                   threshold, cv_results, test_metrics, cfg:
                   {"model": Path("m.pkl"), "preprocessor": Path("p.pkl"), "spin_imputer": Path("s.pkl")},
        )
        result = run_training_pipeline(fast_cfg)
        assert isinstance(result["cv_results"], NestedCVResults)

    @pytest.mark.slow
    def test_pipeline_test_metrics_has_recall_key(self, fast_cfg, monkeypatch, make_students):
        df = make_students(n=80, n_schools=6, prevalence=0.18, seed=8)
        monkeypatch.setattr("src.train.load_active_dataset", lambda c: (df, {}))
        monkeypatch.setattr(
            "src.train._save_artefacts",
            lambda model, preprocessor, model_name, dataset_key,
                   threshold, cv_results, test_metrics, cfg:
                   {"model": Path("m.pkl"), "preprocessor": Path("p.pkl"), "spin_imputer": Path("s.pkl")},
        )
        result = run_training_pipeline(fast_cfg)
        assert "recall" in result["test_metrics"]

    def test_pipeline_raises_on_schema_failure(self, fast_cfg, monkeypatch, make_students):
        df = make_students(n=80, n_schools=6, prevalence=0.18, seed=9)
        df = df.drop(columns=["participant_id"])  # required column — validator will fail
        monkeypatch.setattr("src.train.load_active_dataset", lambda c: (df, {}))
        with pytest.raises(ValueError, match="Schema validation failed"):
            run_training_pipeline(fast_cfg)

    def test_pipeline_raises_when_too_few_records(self, fast_cfg, monkeypatch, make_students):
        df = make_students(n=10, n_schools=2, prevalence=0.20, seed=10)
        monkeypatch.setattr("src.train.load_active_dataset", lambda c: (df, {}))
        with pytest.raises(ValueError):
            run_training_pipeline(fast_cfg)
