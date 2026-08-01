"""Tests for src/explainability.py.

Coverage targets:
  ExplainabilityEngine.__init__            — correct explainer per model type
  ExplainabilityEngine._build_explainer()  — TreeExplainer vs. fallback-to-None
  ExplainabilityEngine.global_shap()       — ranked DataFrame; required columns
  ExplainabilityEngine.explain_instance()  — structured dict; all required keys
  ExplainabilityEngine.surrogate_tree()    — rule string written to disk
  ExplainabilityEngine.export_shap_csv()   — CSV with shap_ columns per feature
  ExplainabilityEngine._prob_to_band()     — correct risk-band thresholds
  ExplainabilityEngine.shap_interaction_plot() — warning for non-TreeExplainer
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import shap

from src.explainability import ExplainabilityEngine


# ── Shared fitted models (module scope — fit once) ───────────────────────────

@pytest.fixture(scope="module")
def feature_names(preprocessed_feature_X) -> list[str]:
    return list(preprocessed_feature_X.columns)


@pytest.fixture(scope="module")
def tree_model(preprocessed_feature_X, feature_y):
    """Shallow DecisionTree — uses shap.TreeExplainer path."""
    m = DecisionTreeClassifier(max_depth=4, random_state=42)
    m.fit(preprocessed_feature_X.values, feature_y.values)
    return m


@pytest.fixture(scope="module")
def lr_model(preprocessed_feature_X, feature_y):
    """LogisticRegression — LinearExplainer fails → explainer set to None."""
    m = LogisticRegression(max_iter=300, random_state=42)
    m.fit(preprocessed_feature_X.values, feature_y.values)
    return m


@pytest.fixture
def tree_engine(cfg, tree_model, feature_names, tmp_path):
    return ExplainabilityEngine(
        cfg, tree_model, feature_names, figures_dir=tmp_path / "figs"
    )


@pytest.fixture
def lr_engine(cfg, lr_model, feature_names, tmp_path):
    return ExplainabilityEngine(
        cfg, lr_model, feature_names, figures_dir=tmp_path / "figs"
    )


# ── ExplainerConstruction ────────────────────────────────────────────────────

class TestExplainerConstruction:
    def test_tree_model_builds_tree_explainer(self, cfg, tree_model, feature_names, tmp_path):
        engine = ExplainabilityEngine(cfg, tree_model, feature_names, figures_dir=tmp_path)
        assert isinstance(engine._explainer, shap.TreeExplainer)

    def test_lr_model_explainer_falls_back_to_none(self, cfg, lr_model, feature_names, tmp_path):
        engine = ExplainabilityEngine(cfg, lr_model, feature_names, figures_dir=tmp_path)
        assert engine._explainer is None

    def test_figures_dir_created_on_init(self, cfg, tree_model, feature_names, tmp_path):
        figs = tmp_path / "new_subdir" / "figs"
        assert not figs.exists()
        ExplainabilityEngine(cfg, tree_model, feature_names, figures_dir=figs)
        assert figs.exists()

    def test_feature_names_stored(self, tree_engine, feature_names):
        assert tree_engine._feature_names == feature_names

    def test_global_shap_values_initially_none(self, tree_engine):
        assert tree_engine._global_shap_values is None


# ── GlobalSHAP ───────────────────────────────────────────────────────────────

class TestGlobalSHAP:
    def test_returns_dataframe(self, tree_engine, preprocessed_feature_X):
        import pandas as pd
        result = tree_engine.global_shap(preprocessed_feature_X, save_plot=False)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.global_shap(preprocessed_feature_X, save_plot=False)
        assert {"feature", "lay_label", "mean_abs_shap"}.issubset(result.columns)

    def test_row_count_equals_n_features(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.global_shap(preprocessed_feature_X, save_plot=False)
        assert len(result) == len(preprocessed_feature_X.columns)

    def test_sorted_by_descending_importance(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.global_shap(preprocessed_feature_X, save_plot=False)
        vals = result["mean_abs_shap"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_mean_abs_shap_all_non_negative(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.global_shap(preprocessed_feature_X, save_plot=False)
        assert (result["mean_abs_shap"] >= 0).all()

    def test_lay_labels_are_strings(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.global_shap(preprocessed_feature_X, save_plot=False)
        assert result["lay_label"].apply(lambda x: isinstance(x, str)).all()

    def test_sets_global_shap_values_cache(self, tree_engine, preprocessed_feature_X):
        tree_engine.global_shap(preprocessed_feature_X, save_plot=False)
        assert tree_engine._global_shap_values is not None


# ── ExplainInstance ──────────────────────────────────────────────────────────

class TestExplainInstance:
    _REQUIRED_KEYS = {
        "risk_probability", "risk_band", "risk_band_label",
        "top_factors", "recommended_action", "model_version",
    }
    _FACTOR_KEYS = {"feature", "lay_label", "shap_value", "direction", "strength"}

    def test_returns_dict_with_required_keys(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.explain_instance(preprocessed_feature_X.iloc[0])
        assert self._REQUIRED_KEYS.issubset(result.keys())

    def test_risk_probability_in_0_1(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.explain_instance(preprocessed_feature_X.iloc[0])
        assert 0.0 <= result["risk_probability"] <= 1.0

    def test_risk_band_is_valid_category(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.explain_instance(preprocessed_feature_X.iloc[0])
        assert result["risk_band"] in {"low", "moderate", "high"}

    def test_default_top_k_yields_three_factors(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.explain_instance(preprocessed_feature_X.iloc[0])
        assert len(result["top_factors"]) == 3

    def test_custom_top_k_respected(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.explain_instance(preprocessed_feature_X.iloc[0], top_k=5)
        assert len(result["top_factors"]) == 5

    def test_each_factor_has_required_keys(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.explain_instance(preprocessed_feature_X.iloc[0])
        for factor in result["top_factors"]:
            assert self._FACTOR_KEYS.issubset(factor.keys())

    def test_direction_is_valid_string(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.explain_instance(preprocessed_feature_X.iloc[0])
        valid = {"increases risk", "decreases risk"}
        for factor in result["top_factors"]:
            assert factor["direction"] in valid

    def test_strength_is_valid_label(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.explain_instance(preprocessed_feature_X.iloc[0])
        valid = {"strong", "moderate", "mild", "negligible"}
        for factor in result["top_factors"]:
            assert factor["strength"] in valid

    def test_accepts_series_input(self, tree_engine, preprocessed_feature_X):
        row_series = preprocessed_feature_X.iloc[0]
        result = tree_engine.explain_instance(row_series)
        assert "risk_probability" in result

    def test_accepts_dataframe_input(self, tree_engine, preprocessed_feature_X):
        row_df = preprocessed_feature_X.iloc[[0]]
        result = tree_engine.explain_instance(row_df)
        assert "risk_probability" in result

    def test_shap_values_in_top_factors_are_floats(self, tree_engine, preprocessed_feature_X):
        result = tree_engine.explain_instance(preprocessed_feature_X.iloc[0])
        for factor in result["top_factors"]:
            assert isinstance(factor["shap_value"], float)


# ── RiskBand mapping ─────────────────────────────────────────────────────────

class TestRiskBand:
    def test_low_band_below_0_30(self, tree_engine):
        assert tree_engine._prob_to_band(0.10) == "low"
        assert tree_engine._prob_to_band(0.00) == "low"
        assert tree_engine._prob_to_band(0.29) == "low"

    def test_moderate_band_at_0_30(self, tree_engine):
        assert tree_engine._prob_to_band(0.30) == "moderate"

    def test_moderate_band_between_0_30_and_0_65(self, tree_engine):
        assert tree_engine._prob_to_band(0.50) == "moderate"
        assert tree_engine._prob_to_band(0.64) == "moderate"

    def test_high_band_at_0_65(self, tree_engine):
        assert tree_engine._prob_to_band(0.65) == "high"

    def test_high_band_above_0_65(self, tree_engine):
        assert tree_engine._prob_to_band(0.90) == "high"
        assert tree_engine._prob_to_band(1.00) == "high"

    def test_band_label_returns_non_empty_string(self, tree_engine):
        for band in ("low", "moderate", "high"):
            label = tree_engine._band_label(band)
            assert isinstance(label, str) and len(label) > 0

    def test_recommended_action_returns_string(self, tree_engine):
        for band in ("low", "moderate", "high"):
            action = tree_engine._recommended_action(band)
            assert isinstance(action, str)


# ── SurrogateTree ────────────────────────────────────────────────────────────

class TestSurrogateTree:
    def test_returns_non_empty_string(self, tree_engine, preprocessed_feature_X):
        y_pred = tree_engine._model.predict(preprocessed_feature_X.values)
        rules = tree_engine.surrogate_tree(preprocessed_feature_X, y_pred)
        assert isinstance(rules, str) and len(rules) > 0

    def test_string_contains_feature_name(self, tree_engine, preprocessed_feature_X):
        y_pred = tree_engine._model.predict(preprocessed_feature_X.values)
        rules = tree_engine.surrogate_tree(preprocessed_feature_X, y_pred)
        feature_names = tree_engine._feature_names
        assert any(fn in rules for fn in feature_names)

    def test_writes_rules_file_to_figures_dir(self, tree_engine, preprocessed_feature_X):
        y_pred = tree_engine._model.predict(preprocessed_feature_X.values)
        tree_engine.surrogate_tree(preprocessed_feature_X, y_pred)
        rules_file = tree_engine._figures_dir / "surrogate_tree_rules.txt"
        assert rules_file.exists()
        assert rules_file.stat().st_size > 0

    def test_max_depth_1_produces_short_output(self, tree_engine, preprocessed_feature_X):
        y_pred = tree_engine._model.predict(preprocessed_feature_X.values)
        rules_deep = tree_engine.surrogate_tree(preprocessed_feature_X, y_pred, max_depth=4)
        rules_shallow = tree_engine.surrogate_tree(preprocessed_feature_X, y_pred, max_depth=1)
        assert len(rules_shallow) < len(rules_deep)


# ── ExportSHAPCsv ────────────────────────────────────────────────────────────

class TestExportSHAPCsv:
    def test_creates_csv_file(self, tree_engine, preprocessed_feature_X, tmp_path):
        out = tmp_path / "shap_out.csv"
        tree_engine.export_shap_csv(preprocessed_feature_X, out)
        assert out.exists()

    def test_csv_row_count_equals_input(self, tree_engine, preprocessed_feature_X, tmp_path):
        import pandas as pd
        out = tmp_path / "shap_out.csv"
        tree_engine.export_shap_csv(preprocessed_feature_X, out)
        df = pd.read_csv(out)
        assert len(df) == len(preprocessed_feature_X)

    def test_csv_columns_prefixed_with_shap_(self, tree_engine, preprocessed_feature_X, tmp_path):
        import pandas as pd
        out = tmp_path / "shap_out.csv"
        tree_engine.export_shap_csv(preprocessed_feature_X, out)
        df = pd.read_csv(out)
        assert all(c.startswith("shap_") for c in df.columns)

    def test_csv_column_count_equals_n_features(self, tree_engine, preprocessed_feature_X, tmp_path):
        import pandas as pd
        out = tmp_path / "shap_out.csv"
        tree_engine.export_shap_csv(preprocessed_feature_X, out)
        df = pd.read_csv(out)
        assert len(df.columns) == len(preprocessed_feature_X.columns)

    def test_creates_parent_dirs(self, tree_engine, preprocessed_feature_X, tmp_path):
        out = tmp_path / "nested" / "path" / "shap.csv"
        assert not out.parent.exists()
        tree_engine.export_shap_csv(preprocessed_feature_X, out)
        assert out.exists()


# ── ShapInteractionPlot ───────────────────────────────────────────────────────

class TestShapInteractionPlot:
    def test_warns_for_non_tree_explainer(self, lr_engine, preprocessed_feature_X, caplog):
        import logging
        feature_names = lr_engine._feature_names
        with caplog.at_level(logging.WARNING):
            lr_engine.shap_interaction_plot(
                preprocessed_feature_X, feature_names[0], feature_names[1]
            )
        assert any("TreeExplainer" in m for m in caplog.messages)

    def test_warns_for_unknown_feature(self, tree_engine, preprocessed_feature_X, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            tree_engine.shap_interaction_plot(
                preprocessed_feature_X, "nonexistent_feature_xyz", "also_nonexistent"
            )
        assert any("not in feature_names" in m or "feature_names" in m for m in caplog.messages)
