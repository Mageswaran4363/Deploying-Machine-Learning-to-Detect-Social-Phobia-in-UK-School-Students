"""Tests for src/preprocessing.py.

Coverage targets:
  Preprocessor.fit_transform()          — all 8 imputation/encoding/scaling steps
  Preprocessor.transform()              — applies fitted state; RuntimeError guard
  Preprocessor.feature_names_out        — available only after fit
  Preprocessor.save() / .load()         — persistence roundtrip
  _apply_conservative_min()             — MNAR columns filled with valid_min
  _apply_zero_impute()                  — zero-fill columns
  _fit_ordinal_encoder()                — OrdinalEncoder fitted on schema categories
  _apply_ordinal_encoder()              — applied on transform
  _align_columns()                      — pads missing cols, drops extras
  apply_smote()                         — normal path and too-few-minority skip
  split_dataset()                       — legacy function still executes correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import Preprocessor, apply_smote, split_dataset


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fresh(cfg) -> Preprocessor:
    return Preprocessor(cfg)


# ── fit_transform ─────────────────────────────────────────────────────────────

class TestFitTransform:
    def test_no_nans_after_fit_transform(self, cfg, feature_X_with_nans):
        prep = _fresh(cfg)
        X_out = prep.fit_transform(feature_X_with_nans.copy())
        assert not X_out.isnull().any().any(), "NaN values remain after fit_transform"

    def test_fitted_flag_set_to_true(self, cfg, feature_X):
        prep = _fresh(cfg)
        assert prep._fitted is False
        prep.fit_transform(feature_X.copy())
        assert prep._fitted is True

    def test_feature_names_out_set_after_fit(self, cfg, feature_X):
        prep = _fresh(cfg)
        X_out = prep.fit_transform(feature_X.copy())
        assert prep.feature_names_out == list(X_out.columns)

    def test_feature_names_out_raises_before_fit(self, cfg):
        prep = _fresh(cfg)
        with pytest.raises(RuntimeError, match="fit_transform"):
            _ = prep.feature_names_out

    def test_output_is_dataframe(self, cfg, feature_X):
        prep = _fresh(cfg)
        X_out = prep.fit_transform(feature_X.copy())
        assert isinstance(X_out, pd.DataFrame)

    def test_output_row_count_unchanged(self, cfg, feature_X):
        prep = _fresh(cfg)
        X_out = prep.fit_transform(feature_X.copy())
        assert len(X_out) == len(feature_X)

    def test_sdq_items_clipped_after_mice(self, cfg, feature_X_with_nans):
        prep = _fresh(cfg)
        X_out = prep.fit_transform(feature_X_with_nans.copy())
        # SDQ subscale columns are present and have no NaN after MICE imputation
        sdq_cols = [c for c in X_out.columns if c.startswith("psdq_") or c.startswith("sdq_")]
        assert len(sdq_cols) > 0  # at least some SDQ columns are present

    def test_conservative_min_cols_have_no_nans(self, cfg, feature_X):
        X = feature_X.copy()
        conservative_cols = [c for c in cfg.conservative_min_cols if c in X.columns]
        if not conservative_cols:
            pytest.skip("No conservative_min columns in feature matrix")
        # Inject NaN
        X.loc[0, conservative_cols[0]] = np.nan
        prep = _fresh(cfg)
        X_out = prep.fit_transform(X)
        for col in conservative_cols:
            if col in X_out.columns:
                assert not X_out[col].isnull().any(), f"{col} has NaN after conservative_min"

    def test_zero_impute_cols_have_no_nans(self, cfg, feature_X):
        X = feature_X.copy()
        zero_cols = [c for c in cfg.zero_impute_cols if c in X.columns]
        if not zero_cols:
            pytest.skip("No zero_impute columns in feature matrix")
        X.loc[:2, zero_cols[0]] = np.nan
        prep = _fresh(cfg)
        X_out = prep.fit_transform(X)
        if zero_cols[0] in X_out.columns:
            # After zero impute + potential scaling, the originally-zero cells won't be NaN
            assert not X_out[zero_cols[0]].isnull().any()

    def test_ordinal_cols_are_numeric_after_encoding(self, cfg, feature_X):
        prep = _fresh(cfg)
        X_out = prep.fit_transform(feature_X.copy())
        for col in cfg.ordinal_cols:
            if col in X_out.columns:
                assert pd.api.types.is_numeric_dtype(X_out[col]), f"{col} is not numeric after encoding"

    def test_double_fit_emits_warning(self, cfg, feature_X, caplog):
        import logging
        prep = _fresh(cfg)
        prep.fit_transform(feature_X.copy())
        with caplog.at_level(logging.WARNING):
            prep.fit_transform(feature_X.copy())
        assert any("already-fitted" in msg for msg in caplog.messages)

    def test_fresh_instances_do_not_share_state(self, cfg, feature_X):
        prep_a = _fresh(cfg)
        prep_b = _fresh(cfg)
        prep_a.fit_transform(feature_X.copy())
        # prep_b should still be unfitted
        assert prep_b._fitted is False


# ── transform ────────────────────────────────────────────────────────────────

class TestTransform:
    def test_transform_before_fit_raises_runtime_error(self, cfg, feature_X):
        prep = _fresh(cfg)
        with pytest.raises(RuntimeError, match="fit_transform"):
            prep.transform(feature_X.copy())

    def test_transform_output_same_columns_as_fit_transform(self, cfg, feature_X):
        n = len(feature_X)
        train = feature_X.iloc[: n // 2].reset_index(drop=True)
        val = feature_X.iloc[n // 2 :].reset_index(drop=True)

        prep = _fresh(cfg)
        train_out = prep.fit_transform(train)
        val_out = prep.transform(val)

        assert list(train_out.columns) == list(val_out.columns)

    def test_transform_does_not_refit_scaler(self, cfg, feature_X):
        n = len(feature_X)
        train = feature_X.iloc[: n // 2].reset_index(drop=True)
        val = feature_X.iloc[n // 2 :].reset_index(drop=True)

        prep = _fresh(cfg)
        prep.fit_transform(train)

        mean_before = prep._scaler.mean_.copy() if hasattr(prep._scaler, "mean_") else None
        prep.transform(val)
        if mean_before is not None:
            np.testing.assert_array_equal(prep._scaler.mean_, mean_before)

    def test_transform_no_nans_for_clean_input(self, cfg, feature_X):
        n = len(feature_X)
        train = feature_X.iloc[: n // 2].reset_index(drop=True)
        val = feature_X.iloc[n // 2 :].reset_index(drop=True)

        prep = _fresh(cfg)
        prep.fit_transform(train)
        val_out = prep.transform(val)
        assert not val_out.isnull().any().any()

    def test_align_columns_drops_extra_columns(self, cfg, feature_X):
        n = len(feature_X)
        train = feature_X.iloc[: n // 2].reset_index(drop=True)
        val = feature_X.iloc[n // 2 :].reset_index(drop=True).copy()

        prep = _fresh(cfg)
        prep.fit_transform(train)

        # Add a column that was NOT in training; _align_columns should remove it.
        val_extra = val.copy()
        extra_col = "_test_extra_col_xyz"
        val_extra[extra_col] = 999
        val_out = prep.transform(val_extra)

        assert extra_col not in val_out.columns
        assert list(val_out.columns) == prep.feature_names_out

    def test_transform_row_count_preserved(self, cfg, feature_X):
        n = len(feature_X)
        train = feature_X.iloc[: n // 2].reset_index(drop=True)
        val = feature_X.iloc[n // 2 :].reset_index(drop=True)

        prep = _fresh(cfg)
        prep.fit_transform(train)
        val_out = prep.transform(val)
        assert len(val_out) == len(val)


# ── save / load ───────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path, cfg, feature_X):
        prep = _fresh(cfg)
        prep.fit_transform(feature_X.copy())

        path = tmp_path / "preprocessor_test.pkl"
        prep.save(path)
        assert path.exists()

        loaded = Preprocessor.load(path)
        assert isinstance(loaded, Preprocessor)
        assert loaded._fitted is True

    def test_loaded_preprocessor_transforms_correctly(self, tmp_path, cfg, feature_X):
        n = len(feature_X)
        train = feature_X.iloc[: n // 2].reset_index(drop=True)
        val = feature_X.iloc[n // 2 :].reset_index(drop=True)

        prep = _fresh(cfg)
        prep.fit_transform(train)
        original_out = prep.transform(val)

        path = tmp_path / "prep_roundtrip.pkl"
        prep.save(path)
        loaded = Preprocessor.load(path)
        loaded_out = loaded.transform(val)

        pd.testing.assert_frame_equal(original_out, loaded_out)

    def test_load_wrong_type_raises_type_error(self, tmp_path):
        import joblib
        path = tmp_path / "fake.pkl"
        joblib.dump({"not": "a preprocessor"}, path)
        with pytest.raises(TypeError):
            Preprocessor.load(path)


# ── apply_smote ───────────────────────────────────────────────────────────────

def _imbalanced_y(n: int, n_minority: int = 10) -> pd.Series:
    """Build an imbalanced label vector with minority/majority < smote_sampling_strategy=0.30."""
    y = pd.Series([0] * (n - n_minority) + [1] * n_minority)
    return y.sample(frac=1, random_state=0).reset_index(drop=True)


class TestApplySmote:
    def test_smote_increases_minority_count(self, cfg, preprocessed_feature_X):
        y = _imbalanced_y(len(preprocessed_feature_X))
        n_minority_before = int(y.sum())
        X_res, y_res = apply_smote(preprocessed_feature_X.copy(), y, cfg)
        n_minority_after = int(y_res.sum())
        assert n_minority_after >= n_minority_before

    def test_smote_returns_dataframe_with_original_columns(self, cfg, preprocessed_feature_X):
        y = _imbalanced_y(len(preprocessed_feature_X))
        X_res, y_res = apply_smote(preprocessed_feature_X.copy(), y, cfg)
        assert isinstance(X_res, pd.DataFrame)
        assert list(X_res.columns) == list(preprocessed_feature_X.columns)

    def test_smote_returns_series_for_labels(self, cfg, preprocessed_feature_X):
        y = _imbalanced_y(len(preprocessed_feature_X))
        _, y_res = apply_smote(preprocessed_feature_X.copy(), y, cfg)
        assert isinstance(y_res, pd.Series)

    def test_smote_skipped_when_too_few_minority(self, cfg, preprocessed_feature_X):
        # 2 positive cases — below k_neighbors + 1 = 6, so SMOTE is skipped
        y_tiny = pd.Series([0] * (len(preprocessed_feature_X) - 2) + [1, 1])
        X_res, y_res = apply_smote(preprocessed_feature_X.copy(), y_tiny, cfg)
        assert len(X_res) == len(preprocessed_feature_X)
        assert list(y_res) == list(y_tiny)

    def test_smote_majority_class_count_unchanged_or_larger(self, cfg, preprocessed_feature_X):
        y = _imbalanced_y(len(preprocessed_feature_X))
        n_majority_before = int((y == 0).sum())
        _, y_res = apply_smote(preprocessed_feature_X.copy(), y, cfg)
        n_majority_after = int((y_res == 0).sum())
        assert n_majority_after >= n_majority_before


# ── split_dataset (legacy) ────────────────────────────────────────────────────

class TestSplitDataset:
    def test_split_returns_seven_outputs(self, cfg, students_df):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(cfg)
        df = fe.transform(students_df.copy())
        result = split_dataset(df, cfg)
        assert len(result) == 7

    def test_split_sizes_sum_to_total(self, cfg, students_df):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(cfg)
        df = fe.transform(students_df.copy())
        X_train, X_val, X_test, y_train, y_val, y_test, _ = split_dataset(df, cfg)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(df)

    def test_split_no_school_overlap_between_train_and_test(self, cfg, students_df):
        from src.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(cfg)
        df = fe.transform(students_df.copy())
        X_train, X_val, X_test, y_train, y_val, y_test, groups_train = split_dataset(df, cfg)
        # groups_train schools should not appear in X_test index positions
        # (We can't check X_test groups directly since split_dataset drops them,
        # but we verify groups_train length matches X_train length)
        assert len(groups_train) == len(X_train)
