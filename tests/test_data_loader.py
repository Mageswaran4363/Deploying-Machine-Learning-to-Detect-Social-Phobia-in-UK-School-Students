"""Tests for src/data_loader.py.

Coverage targets:
  _build_dtype_map()               — schema-driven dtype dict
  load_active_dataset()            — FileNotFoundError when CSV absent; correct (df, meta) tuple
  load_dataset_by_key()            — key swap; active key restored; raises on missing file
  apply_exclusion_criteria()       — consent filter; quality flag; SPIN missingness; on-the-fly count
  dataset_registry.yaml            — active key; expected dataset entries; metadata fields
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data_loader import (
    _build_dtype_map,
    apply_exclusion_criteria,
    load_active_dataset,
    load_dataset_by_key,
)

PARENT_SDQ_ITEMS = ["psdq_emotional", "psdq_peer", "psdq_conduct", "psdq_hyper", "psdq_prosocial"]
SELF_SDQ_ITEMS   = ["sdq_emotional", "sdq_peer", "sdq_conduct", "sdq_hyper", "sdq_prosocial"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fresh_cfg() -> Config:
    """Function-scoped Config so registry mutations don't bleed between tests."""
    return Config()


@pytest.fixture
def cfg_with_test_csv(tmp_path) -> tuple:
    """Config pointing real_v1 at a small but fully valid test CSV (schema v2.0).

    Uses conftest._make_students() which generates SDQ-based schema-compliant data.
    """
    from tests.conftest import _make_students
    cfg = Config()
    df = _make_students(n=30, n_schools=5, seed=99)
    csv_path = tmp_path / "test_real.csv"
    df.to_csv(csv_path, index=False)
    cfg._dataset_registry["datasets"]["real_v1"]["path"] = str(csv_path)
    return cfg, df


# ── _build_dtype_map ──────────────────────────────────────────────────────────

class TestBuildDtypeMap:
    def test_returns_dict(self, fresh_cfg):
        dtype_map = _build_dtype_map(fresh_cfg)
        assert isinstance(dtype_map, dict)

    def test_keys_are_schema_column_names(self, fresh_cfg):
        dtype_map = _build_dtype_map(fresh_cfg)
        schema_names = {col["name"] for col in fresh_cfg._columns}
        assert set(dtype_map.keys()) == schema_names

    def test_string_columns_map_to_object(self, fresh_cfg):
        dtype_map = _build_dtype_map(fresh_cfg)
        # participant_id is a string column
        assert dtype_map.get("participant_id") == "object"

    def test_integer_columns_map_to_Int64(self, fresh_cfg):
        dtype_map = _build_dtype_map(fresh_cfg)
        # spin_risk_binary is an integer column
        assert dtype_map.get("spin_risk_binary") == "Int64"

    def test_float_columns_map_to_Float64(self, fresh_cfg):
        dtype_map = _build_dtype_map(fresh_cfg)
        # wellbeing_wemwbs is the float column in schema v2.0
        assert dtype_map.get("wellbeing_wemwbs") == "Float64"

    def test_sdq_columns_are_present(self, fresh_cfg):
        dtype_map = _build_dtype_map(fresh_cfg)
        for col in ["psdq_emotional", "sdq_emotional", "wellbeing_wemwbs"]:
            assert col in dtype_map


# ── load_active_dataset ───────────────────────────────────────────────────────

class TestLoadActiveDataset:
    def test_raises_file_not_found_when_csv_absent(self, fresh_cfg, tmp_path, monkeypatch):
        # Patch the path inside the registry dict — avoids monkeypatching a property
        monkeypatch.setitem(
            fresh_cfg._dataset_registry["datasets"]["real_v1"],
            "path",
            str(tmp_path / "nonexistent.csv"),
        )
        with pytest.raises(FileNotFoundError):
            load_active_dataset(fresh_cfg)

    def test_returns_tuple_of_df_and_dict(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        result = load_active_dataset(cfg)
        assert isinstance(result, tuple) and len(result) == 2
        df, meta = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(meta, dict)

    def test_df_has_rows(self, cfg_with_test_csv):
        cfg, source_df = cfg_with_test_csv
        df, _ = load_active_dataset(cfg)
        assert len(df) == len(source_df)

    def test_df_has_sdq_item_columns(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        df, _ = load_active_dataset(cfg)
        for col in PARENT_SDQ_ITEMS + SELF_SDQ_ITEMS:
            assert col in df.columns, f"Missing SDQ column: {col}"

    def test_meta_has_label_key(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        _, meta = load_active_dataset(cfg)
        assert "label" in meta

    def test_meta_has_path_key(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        _, meta = load_active_dataset(cfg)
        assert "path" in meta

    def test_df_has_participant_id_column(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        df, _ = load_active_dataset(cfg)
        assert "participant_id" in df.columns

    def test_df_has_school_id_column(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        df, _ = load_active_dataset(cfg)
        assert "school_id" in df.columns

    def test_wellbeing_wemwbs_is_float_dtype(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        df, _ = load_active_dataset(cfg)
        assert pd.api.types.is_float_dtype(df["wellbeing_wemwbs"])

    def test_psdq_emotional_is_integer_dtype(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        df, _ = load_active_dataset(cfg)
        assert pd.api.types.is_integer_dtype(df["psdq_emotional"])


# ── load_dataset_by_key ───────────────────────────────────────────────────────

class TestLoadDatasetByKey:
    def test_loads_real_v1_returns_dataframe(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        df, _ = load_dataset_by_key(cfg, "real_v1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_restores_active_key_after_successful_load(self, cfg_with_test_csv):
        cfg, _ = cfg_with_test_csv
        original = cfg._dataset_registry["active"]
        load_dataset_by_key(cfg, "real_v1")
        assert cfg._dataset_registry["active"] == original

    def test_restores_active_key_after_exception(self, fresh_cfg):
        original = fresh_cfg._dataset_registry["active"]
        with pytest.raises(FileNotFoundError):
            load_dataset_by_key(fresh_cfg, "pilot_v1")  # file does not exist
        assert fresh_cfg._dataset_registry["active"] == original

    def test_raises_file_not_found_for_missing_dataset(self, fresh_cfg):
        with pytest.raises(FileNotFoundError):
            load_dataset_by_key(fresh_cfg, "pilot_v1")


# ── apply_exclusion_criteria ──────────────────────────────────────────────────

class TestApplyExclusionCriteria:
    def _base_df(self, n: int = 20) -> pd.DataFrame:
        """Minimal DataFrame with all required exclusion columns (schema v2.0)."""
        rng = np.random.default_rng(0)
        data = {
            "participant_id": [f"P{i:04d}" for i in range(n)],
            "spin_items_missing": [0] * n,
            "consent_obtained": [1] * n,
            "data_quality_flag": [0] * n,
        }
        for col in PARENT_SDQ_ITEMS:
            data[col] = rng.integers(0, 11, size=n).tolist()
        return pd.DataFrame(data)

    def test_no_exclusion_cols_keeps_all_rows(self, fresh_cfg):
        df = pd.DataFrame({"x": range(30)})
        out = apply_exclusion_criteria(df, fresh_cfg)
        assert len(out) == 30

    def test_consent_filter_removes_non_consented(self, fresh_cfg):
        df = self._base_df(20)
        df.loc[0:4, "consent_obtained"] = 0  # 5 without consent
        out = apply_exclusion_criteria(df, fresh_cfg)
        assert len(out) == 15

    def test_quality_flag_removes_flagged_rows(self, fresh_cfg):
        df = self._base_df(20)
        df.loc[0:2, "data_quality_flag"] = 1  # 3 flagged
        out = apply_exclusion_criteria(df, fresh_cfg)
        assert len(out) == 17

    def test_spin_missing_filter_removes_excessive(self, fresh_cfg):
        df = self._base_df(20)
        max_ok = fresh_cfg.max_missing_spin_items
        df.loc[0, "spin_items_missing"] = max_ok + 1  # exactly 1 over
        out = apply_exclusion_criteria(df, fresh_cfg)
        assert len(out) == 19

    def test_all_criteria_combined(self, fresh_cfg):
        df = self._base_df(30)
        max_ok = fresh_cfg.max_missing_spin_items
        df.loc[0:2, "consent_obtained"] = 0     # 3 removed
        df.loc[3:4, "data_quality_flag"] = 1    # 2 removed
        df.loc[5, "spin_items_missing"] = max_ok + 2  # 1 removed
        out = apply_exclusion_criteria(df, fresh_cfg)
        assert len(out) == 24

    def test_computes_sdq_missingness_on_the_fly_when_column_absent(self, fresh_cfg):
        df = self._base_df(20)
        df = df.drop(columns=["spin_items_missing"])
        # Inject NaN into > max_missing parent SDQ items for first row
        max_ok = fresh_cfg.max_missing_spin_items
        for col in PARENT_SDQ_ITEMS[:max_ok + 1]:
            df.loc[0, col] = np.nan
        out = apply_exclusion_criteria(df, fresh_cfg)
        assert len(out) == 19

    def test_resets_index_after_filtering(self, fresh_cfg):
        df = self._base_df(10)
        df.loc[0, "consent_obtained"] = 0
        out = apply_exclusion_criteria(df, fresh_cfg)
        assert list(out.index) == list(range(len(out)))

    def test_all_rows_pass_when_fully_compliant(self, fresh_cfg):
        df = self._base_df(15)
        out = apply_exclusion_criteria(df, fresh_cfg)
        assert len(out) == 15

    def test_nan_consent_treated_as_not_consented(self, fresh_cfg):
        df = self._base_df(10)
        df.loc[0, "consent_obtained"] = np.nan
        out = apply_exclusion_criteria(df, fresh_cfg)
        assert len(out) == 9


# ── Dataset registry ──────────────────────────────────────────────────────────

class TestDatasetRegistry:
    def test_active_key_is_real_v1(self, fresh_cfg):
        assert fresh_cfg.active_dataset_key == "real_v1"

    def test_registry_contains_synthetic_v1(self, fresh_cfg):
        assert "synthetic_v1" in fresh_cfg._dataset_registry["datasets"]

    def test_registry_contains_pilot_v1(self, fresh_cfg):
        assert "pilot_v1" in fresh_cfg._dataset_registry["datasets"]

    def test_registry_contains_real_v1(self, fresh_cfg):
        assert "real_v1" in fresh_cfg._dataset_registry["datasets"]

    def test_synthetic_v1_is_marked_validated(self, fresh_cfg):
        meta = fresh_cfg._dataset_registry["datasets"]["synthetic_v1"]
        assert meta.get("validated") is True

    def test_pilot_v1_is_not_validated(self, fresh_cfg):
        meta = fresh_cfg._dataset_registry["datasets"]["pilot_v1"]
        assert meta.get("validated") is not True

    def test_real_v1_is_validated(self, fresh_cfg):
        meta = fresh_cfg._dataset_registry["datasets"]["real_v1"]
        assert meta.get("validated") is True

    def test_synthetic_v1_path_points_to_csv(self, fresh_cfg):
        path_str = fresh_cfg._dataset_registry["datasets"]["synthetic_v1"]["path"]
        assert path_str.endswith(".csv")

    def test_active_dataset_meta_matches_real_v1(self, fresh_cfg):
        meta = fresh_cfg.active_dataset
        assert meta == fresh_cfg._dataset_registry["datasets"]["real_v1"]
