"""Tests for src/feature_engineering.py (schema v2.0 — SDQ/WEMWBS features).

Coverage targets:
  FeatureEngineer.transform()             — all sub-steps in one call
  FeatureEngineer.engineered_col_names()  — names list
  _compute_sdq_internalising()            — sdq_internalising, psdq_internalising
  _compute_parent_self_discrepancy()      — parent_self_discrepancy
  _compute_wellbeing_flag()               — wellbeing_low_flag threshold
  _compute_any_bullying()                 — composite flag; NaN propagation rules
  Graceful degradation                    — missing source columns produce no crash
  Statelessness                           — no fit(); same row → same result in any batch
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FeatureEngineer, _WEMWBS_LOW_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _one_row(**overrides) -> pd.DataFrame:
    """Single-row DataFrame with all schema v2.0 columns populated."""
    data = {
        # Parent SDQ subscales (0–10)
        "psdq_emotional": [5], "psdq_peer": [4], "psdq_conduct": [2],
        "psdq_hyper": [3],     "psdq_prosocial": [7],
        "psdq_total": [21],   # 5+4+2+3+7
        # Self SDQ subscales (0–10)
        "sdq_emotional": [3], "sdq_peer": [3], "sdq_conduct": [2],
        "sdq_hyper": [4],     "sdq_prosocial": [8],
        "sdq_total": [20],    # 3+3+2+4+8
        # WEMWBS
        "wellbeing_wemwbs": [48.0],
        # Bullying
        "bullied_flag": [0],
        "cyberbullied_flag": [0],
    }
    data.update(overrides)
    return pd.DataFrame(data)


@pytest.fixture
def fe(cfg) -> FeatureEngineer:
    return FeatureEngineer(cfg)


# ── Engineered column names ───────────────────────────────────────────────────

class TestEngineeredColNames:
    def test_returns_list_of_strings(self, cfg):
        fe = FeatureEngineer(cfg)
        names = fe.engineered_col_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_contains_sdq_internalising(self, cfg):
        names = FeatureEngineer(cfg).engineered_col_names()
        assert "sdq_internalising" in names

    def test_contains_psdq_internalising(self, cfg):
        names = FeatureEngineer(cfg).engineered_col_names()
        assert "psdq_internalising" in names

    def test_contains_parent_self_discrepancy(self, cfg):
        names = FeatureEngineer(cfg).engineered_col_names()
        assert "parent_self_discrepancy" in names

    def test_contains_wellbeing_low_flag(self, cfg):
        names = FeatureEngineer(cfg).engineered_col_names()
        assert "wellbeing_low_flag" in names

    def test_contains_any_bullying(self, cfg):
        names = FeatureEngineer(cfg).engineered_col_names()
        assert "any_bullying" in names

    def test_all_names_in_transform_output(self, fe):
        df = _one_row()
        result = fe.transform(df)
        for col in fe.engineered_col_names():
            assert col in result.columns, f"Missing engineered column: {col}"


# ── SDQ internalising composites ─────────────────────────────────────────────

class TestSDQInternalising:
    def test_sdq_internalising_equals_emotional_plus_peer(self, fe):
        df = _one_row(sdq_emotional=[5], sdq_peer=[3])
        result = fe.transform(df)
        assert result.loc[0, "sdq_internalising"] == pytest.approx(8.0)

    def test_psdq_internalising_equals_parent_emotional_plus_peer(self, fe):
        df = _one_row(psdq_emotional=[6], psdq_peer=[4])
        result = fe.transform(df)
        assert result.loc[0, "psdq_internalising"] == pytest.approx(10.0)

    def test_sdq_internalising_max_is_20(self, fe):
        df = _one_row(sdq_emotional=[10], sdq_peer=[10])
        result = fe.transform(df)
        assert result.loc[0, "sdq_internalising"] == pytest.approx(20.0)

    def test_sdq_internalising_min_is_0(self, fe):
        df = _one_row(sdq_emotional=[0], sdq_peer=[0])
        result = fe.transform(df)
        assert result.loc[0, "sdq_internalising"] == pytest.approx(0.0)

    def test_sdq_internalising_range_bounded(self, fe, students_df):
        result = fe.transform(students_df.copy())
        col = result["sdq_internalising"].dropna()
        assert (col >= 0).all()
        assert (col <= 20).all()

    def test_psdq_internalising_range_bounded(self, fe, students_df):
        result = fe.transform(students_df.copy())
        col = result["psdq_internalising"].dropna()
        assert (col >= 0).all()
        assert (col <= 20).all()

    def test_preexisting_sdq_internalising_not_overwritten(self, fe):
        df = _one_row()
        df["sdq_internalising"] = 999
        result = fe.transform(df)
        assert result.loc[0, "sdq_internalising"] == pytest.approx(999.0)

    def test_nan_propagated_when_source_is_nan(self, fe):
        df = _one_row(sdq_emotional=[np.nan])
        result = fe.transform(df)
        assert pd.isna(result.loc[0, "sdq_internalising"])


# ── Parent–self discrepancy ───────────────────────────────────────────────────

class TestParentSelfDiscrepancy:
    def test_discrepancy_formula_parent_minus_self(self, fe):
        df = _one_row(psdq_total=[25], sdq_total=[20])
        result = fe.transform(df)
        assert result.loc[0, "parent_self_discrepancy"] == pytest.approx(5.0)

    def test_discrepancy_negative_when_child_reports_more(self, fe):
        df = _one_row(psdq_total=[15], sdq_total=[22])
        result = fe.transform(df)
        assert result.loc[0, "parent_self_discrepancy"] == pytest.approx(-7.0)

    def test_discrepancy_zero_when_equal(self, fe):
        df = _one_row(psdq_total=[20], sdq_total=[20])
        result = fe.transform(df)
        assert result.loc[0, "parent_self_discrepancy"] == pytest.approx(0.0)

    def test_preexisting_discrepancy_not_overwritten(self, fe):
        df = _one_row()
        df["parent_self_discrepancy"] = 777
        result = fe.transform(df)
        assert result.loc[0, "parent_self_discrepancy"] == pytest.approx(777.0)

    def test_missing_psdq_total_skips_feature(self, fe):
        df = _one_row()
        df = df.drop(columns=["psdq_total"])
        result = fe.transform(df)
        assert "parent_self_discrepancy" not in result.columns

    def test_missing_sdq_total_skips_feature(self, fe):
        df = _one_row()
        df = df.drop(columns=["sdq_total"])
        result = fe.transform(df)
        assert "parent_self_discrepancy" not in result.columns


# ── Wellbeing flag ────────────────────────────────────────────────────────────

class TestWellbeingFlag:
    def test_flag_1_when_wemwbs_below_threshold(self, fe):
        df = _one_row(wellbeing_wemwbs=[_WEMWBS_LOW_THRESHOLD - 1.0])
        result = fe.transform(df)
        assert result.loc[0, "wellbeing_low_flag"] == pytest.approx(1.0)

    def test_flag_0_when_wemwbs_above_threshold(self, fe):
        df = _one_row(wellbeing_wemwbs=[_WEMWBS_LOW_THRESHOLD + 1.0])
        result = fe.transform(df)
        assert result.loc[0, "wellbeing_low_flag"] == pytest.approx(0.0)

    def test_flag_0_at_exact_threshold(self, fe):
        # threshold condition is strict < so exactly at threshold → 0
        df = _one_row(wellbeing_wemwbs=[_WEMWBS_LOW_THRESHOLD])
        result = fe.transform(df)
        assert result.loc[0, "wellbeing_low_flag"] == pytest.approx(0.0)

    def test_nan_propagated_when_wemwbs_is_nan(self, fe):
        df = _one_row(wellbeing_wemwbs=[np.nan])
        result = fe.transform(df)
        assert pd.isna(result.loc[0, "wellbeing_low_flag"])

    def test_flag_bounded_0_to_1(self, fe, students_df):
        result = fe.transform(students_df.copy())
        col = result["wellbeing_low_flag"].dropna()
        assert set(col.unique()).issubset({0.0, 1.0})

    def test_preexisting_flag_not_overwritten(self, fe):
        df = _one_row()
        df["wellbeing_low_flag"] = 999
        result = fe.transform(df)
        assert result.loc[0, "wellbeing_low_flag"] == pytest.approx(999.0)

    def test_missing_wemwbs_column_skips_feature(self, fe):
        df = _one_row()
        df = df.drop(columns=["wellbeing_wemwbs"])
        result = fe.transform(df)
        assert "wellbeing_low_flag" not in result.columns


# ── Any bullying flag ─────────────────────────────────────────────────────────

class TestAnyBullying:
    def test_flag_1_when_school_bullied(self, fe):
        df = _one_row(bullied_flag=[1], cyberbullied_flag=[0])
        result = fe.transform(df)
        assert result.loc[0, "any_bullying"] == pytest.approx(1.0)

    def test_flag_1_when_cyberbullied(self, fe):
        df = _one_row(bullied_flag=[0], cyberbullied_flag=[1])
        result = fe.transform(df)
        assert result.loc[0, "any_bullying"] == pytest.approx(1.0)

    def test_flag_1_when_both_bullied(self, fe):
        df = _one_row(bullied_flag=[1], cyberbullied_flag=[1])
        result = fe.transform(df)
        assert result.loc[0, "any_bullying"] == pytest.approx(1.0)

    def test_flag_0_when_neither(self, fe):
        df = _one_row(bullied_flag=[0], cyberbullied_flag=[0])
        result = fe.transform(df)
        assert result.loc[0, "any_bullying"] == pytest.approx(0.0)

    def test_flag_1_when_only_cyber_source_present(self, fe):
        df = _one_row(cyberbullied_flag=[1])
        df = df.drop(columns=["bullied_flag"])
        result = fe.transform(df)
        assert result.loc[0, "any_bullying"] == pytest.approx(1.0)

    def test_nan_when_both_sources_nan(self, fe):
        df = _one_row(bullied_flag=[np.nan], cyberbullied_flag=[np.nan])
        result = fe.transform(df)
        assert pd.isna(result.loc[0, "any_bullying"])

    def test_school_bullied_overrides_cyber_nan(self, fe):
        df = _one_row(bullied_flag=[1], cyberbullied_flag=[np.nan])
        result = fe.transform(df)
        assert result.loc[0, "any_bullying"] == pytest.approx(1.0)

    def test_missing_both_columns_skips_feature(self, fe):
        df = _one_row()
        df = df.drop(columns=["bullied_flag", "cyberbullied_flag"])
        result = fe.transform(df)
        assert "any_bullying" not in result.columns

    def test_flag_bounded_0_to_1(self, fe, students_df):
        result = fe.transform(students_df.copy())
        col = result["any_bullying"].dropna()
        assert set(col.unique()).issubset({0.0, 1.0})


# ── Statelessness ─────────────────────────────────────────────────────────────

class TestStatelessness:
    def test_no_fit_method_on_class(self):
        assert not hasattr(FeatureEngineer, "fit"), \
            "FeatureEngineer must not have a fit() method — it is stateless"

    def test_same_row_same_result_in_any_batch(self, fe):
        row = _one_row()
        result_single = fe.transform(row.copy())
        batch = pd.concat([row, row, row], ignore_index=True)
        result_batch = fe.transform(batch)
        pd.testing.assert_series_equal(
            result_batch.iloc[0].reset_index(drop=True),
            result_single.iloc[0].reset_index(drop=True),
        )

    def test_output_row_count_equals_input(self, fe, students_df):
        result = fe.transform(students_df.copy())
        assert len(result) == len(students_df)

    def test_transform_adds_columns_to_existing_df(self, fe):
        df = _one_row()
        n_cols_before = df.shape[1]
        result = fe.transform(df)
        assert result.shape[1] > n_cols_before

    def test_transform_does_not_mutate_input(self, fe):
        df = _one_row()
        original_cols = set(df.columns)
        fe.transform(df)
        assert set(df.columns) == original_cols, "transform() must not mutate the input DataFrame"

    def test_transform_on_full_students_df_adds_all_engineered_cols(self, fe, students_df):
        result = fe.transform(students_df.copy())
        for col in fe.engineered_col_names():
            assert col in result.columns
