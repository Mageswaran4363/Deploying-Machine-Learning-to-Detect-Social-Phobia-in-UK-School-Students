"""Shared fixtures for the social-phobia-research test suite.

Build order:
  _make_students()        → raw schema v2.0-compliant DataFrame (SDQ + demographics; no derived cols)
  fe.transform(raw_df)    → adds sdq_internalising, psdq_internalising, parent_self_discrepancy,
                            wellbeing_low_flag, any_bullying
  X = df[model_feature_cols] → feature matrix fed to Preprocessor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config

# --------------------------------------------------------------------------- #
# Constants derived from schema v2.0                                           #
# --------------------------------------------------------------------------- #

PARENT_SDQ_ITEMS  = ["psdq_emotional", "psdq_peer", "psdq_conduct", "psdq_hyper", "psdq_prosocial"]
SELF_SDQ_ITEMS    = ["sdq_emotional",  "sdq_peer",  "sdq_conduct",  "sdq_hyper",  "sdq_prosocial"]
MICE_COLS         = PARENT_SDQ_ITEMS + SELF_SDQ_ITEMS + ["wellbeing_wemwbs"]


# --------------------------------------------------------------------------- #
# Raw-data factory                                                             #
# --------------------------------------------------------------------------- #

def _make_students(
    n: int = 100,
    n_schools: int = 8,
    prevalence: float = 0.10,   # higher than real (2%) so SMOTE gets enough minority cases
    seed: int = 42,
) -> pd.DataFrame:
    """Return a schema v2.0-compliant raw DataFrame — no engineered derived columns.

    At-risk students (spin_risk_binary=1) have elevated SDQ emotional/peer scores
    and lower WEMWBS wellbeing, matching the expected signal in real UKDA data.
    All columns are fully populated (no NaN) — NaN injection for imputation tests
    is done separately in feature_X_with_nans.
    """
    rng = np.random.default_rng(seed)
    n_at_risk = max(int(n * prevalence), 6)   # at least 6 for SMOTE k=5
    schools = [f"SCH{i:03d}" for i in range(1, n_schools + 1)]

    at_risk = np.zeros(n, dtype=int)
    at_risk[:n_at_risk] = 1
    rng.shuffle(at_risk)

    rows: dict = {
        # Identifiers
        "participant_id": [f"P{i:05d}" for i in range(n)],
        "school_id":      rng.choice(schools, size=n).tolist(),
        # Metadata
        "cohort_wave":        [2017] * n,
        "consent_obtained":   [1] * n,
        "data_quality_flag":  [0] * n,
        "spin_items_missing": [0] * n,
        "spin_complete_flag": [1] * n,
        # Target (ICD-10 social phobia proxy — at-risk flag)
        "spin_risk_binary": at_risk.tolist(),
        # Demographics
        "age_years":              rng.integers(11, 19, size=n).tolist(),
        "gender":                 rng.choice(["male", "female"], size=n).tolist(),
        "ethnicity_bme":          rng.choice([0, 1], p=[0.75, 0.25], size=n).tolist(),
        "deprivation_quintile":   rng.choice([1, 2, 3, 4, 5], size=n).tolist(),
        "sen_status":             rng.choice([0, 1], p=[0.80, 0.20], size=n).tolist(),
        "learning_difficulty_level": rng.choice([1, 2, 3, 4], p=[0.55, 0.30, 0.10, 0.05], size=n).tolist(),
    }

    # Parent SDQ: at-risk have higher emotional/peer, lower prosocial
    for col in PARENT_SDQ_ITEMS:
        if col in ("psdq_emotional", "psdq_peer"):
            vals = np.where(
                at_risk == 1,
                np.clip(rng.integers(4, 11, size=n), 0, 10),
                np.clip(rng.integers(0,  6, size=n), 0, 10),
            )
        elif col == "psdq_prosocial":
            vals = np.where(
                at_risk == 1,
                np.clip(rng.integers(3,  8, size=n), 0, 10),
                np.clip(rng.integers(6, 11, size=n), 0, 10),
            )
        else:
            vals = np.clip(rng.integers(0, 8, size=n), 0, 10)
        rows[col] = vals.tolist()

    psdq_items = np.column_stack([rows[c] for c in PARENT_SDQ_ITEMS])
    rows["psdq_total"] = psdq_items.sum(axis=1).clip(0, 40).tolist()

    # Self SDQ: same pattern
    for col in SELF_SDQ_ITEMS:
        if col in ("sdq_emotional", "sdq_peer"):
            vals = np.where(
                at_risk == 1,
                np.clip(rng.integers(4, 11, size=n), 0, 10),
                np.clip(rng.integers(0,  5, size=n), 0, 10),
            )
        elif col == "sdq_prosocial":
            vals = np.where(
                at_risk == 1,
                np.clip(rng.integers(3,  8, size=n), 0, 10),
                np.clip(rng.integers(6, 11, size=n), 0, 10),
            )
        else:
            vals = np.clip(rng.integers(0, 7, size=n), 0, 10)
        rows[col] = vals.tolist()

    sdq_items = np.column_stack([rows[c] for c in SELF_SDQ_ITEMS])
    rows["sdq_total"] = sdq_items.sum(axis=1).clip(0, 40).tolist()

    # WEMWBS: at-risk have lower wellbeing
    rows["wellbeing_wemwbs"] = np.where(
        at_risk == 1,
        np.clip(rng.normal(38.0, 8.0, size=n), 14.0, 70.0),
        np.clip(rng.normal(53.0, 8.0, size=n), 14.0, 70.0),
    ).round(1).tolist()

    # Bullying
    rows["bullied_flag"] = np.where(
        at_risk == 1,
        rng.choice([0, 1], p=[0.40, 0.60], size=n),
        rng.choice([0, 1], p=[0.70, 0.30], size=n),
    ).tolist()
    rows["cyberbullied_flag"] = rng.choice([0, 1], p=[0.75, 0.25], size=n).tolist()

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Core fixtures                                                                #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config()


@pytest.fixture
def make_students():
    """Factory fixture: call with keyword args to override defaults."""
    return _make_students


@pytest.fixture(scope="session")
def students_df() -> pd.DataFrame:
    """100 students across 8 schools; schema v2.0-compliant; no NaN."""
    return _make_students(n=100, n_schools=8, prevalence=0.10, seed=42)


# --------------------------------------------------------------------------- #
# Preprocessor-specific fixtures                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def feature_X(cfg, students_df) -> pd.DataFrame:
    """Feature matrix ready for Preprocessor.fit_transform().

    Applies FeatureEngineer first (adds sdq_internalising etc.),
    then filters to schema-defined model feature columns.
    """
    from src.feature_engineering import FeatureEngineer
    fe = FeatureEngineer(cfg)
    df = fe.transform(students_df.copy())
    feature_cols = [c for c in cfg.model_feature_cols if c in df.columns]
    return df[feature_cols].copy()


@pytest.fixture(scope="session")
def feature_y(students_df, cfg) -> pd.Series:
    return students_df[cfg.primary_target_col].astype(int)


@pytest.fixture(scope="session")
def feature_groups(students_df, cfg) -> pd.Series:
    return students_df[cfg.group_col]


@pytest.fixture
def feature_X_with_nans(feature_X) -> pd.DataFrame:
    """Feature matrix with NaN injected into SDQ subscale and wellbeing columns."""
    X = feature_X.copy()
    rng = np.random.default_rng(7)
    sdq_cols = [c for c in X.columns if c in MICE_COLS]
    for col in sdq_cols:
        mask = rng.random(len(X)) < 0.10
        X.loc[mask, col] = np.nan
    return X


@pytest.fixture(scope="session")
def preprocessed_feature_X(cfg, feature_X) -> pd.DataFrame:
    """feature_X after Preprocessor.fit_transform() — fully numeric, no NaN.

    Used by SMOTE tests, which require all-numeric input.
    """
    from src.preprocessing import Preprocessor
    prep = Preprocessor(cfg)
    return prep.fit_transform(feature_X.copy())
