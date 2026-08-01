"""
Generate synthetic_v1.csv — a reproducible synthetic dataset for pipeline
development and testing (schema v2.0: SDQ + WEMWBS).

Design
------
* N=300 students, 10 schools, 10% at-risk prevalence (~30 positive cases).
* Column names, valid ranges, and categories sourced from config/schema.json.
* At-risk students have elevated SDQ emotional/peer scores and lower WEMWBS,
  matching the expected signal in real UKDA-8467 data.
* ~8% of SDQ/WEMWBS values are randomly set to NaN to exercise MICE imputation.

Usage
-----
    python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEED       = 42
N          = 300
N_SCHOOLS  = 10
PREVALENCE = 0.10
MISS_RATE  = 0.08   # SDQ / WEMWBS missingness rate


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _add_missing(arr: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    arr = arr.astype(float).copy()
    arr[rng.random(len(arr)) < rate] = np.nan
    return arr


def generate(cfg: Config, seed: int = SEED, n: int = N) -> pd.DataFrame:
    """Generate a schema v2.0-compliant synthetic DataFrame."""
    rng = _rng(seed)

    n_at_risk = max(int(n * PREVALENCE), 6)
    school_ids = [f"SCH{i:03d}" for i in range(1, N_SCHOOLS + 1)]
    schools = rng.choice(school_ids, size=n, replace=True)

    at_risk = np.zeros(n, dtype=int)
    at_risk[:n_at_risk] = 1
    rng.shuffle(at_risk)

    rows: dict = {}

    # ── Identifiers ──────────────────────────────────────────────────────────
    rows["participant_id"]    = [f"P{i:05d}" for i in range(n)]
    rows["school_id"]         = list(schools)
    rows["cohort_wave"]       = [2017] * n

    # ── Metadata / QC ────────────────────────────────────────────────────────
    rows["consent_obtained"]   = [1] * n
    rows["data_quality_flag"]  = [0] * n
    rows["spin_items_missing"] = [0] * n
    rows["spin_complete_flag"] = [1] * n

    # ── Target ───────────────────────────────────────────────────────────────
    rows["spin_risk_binary"] = at_risk.tolist()

    # ── Demographics ─────────────────────────────────────────────────────────
    rows["age_years"]          = rng.integers(11, 19, size=n).tolist()
    rows["gender"]             = rng.choice(["male", "female"], size=n).tolist()
    rows["ethnicity_bme"]      = rng.choice([0, 1], p=[0.75, 0.25], size=n).tolist()
    rows["deprivation_quintile"] = rng.choice([1, 2, 3, 4, 5], size=n).tolist()
    rows["sen_status"]         = rng.choice([0, 1], p=[0.80, 0.20], size=n).tolist()
    rows["learning_difficulty_level"] = rng.choice(
        [1, 2, 3, 4], p=[0.55, 0.30, 0.10, 0.05], size=n
    ).tolist()

    # ── Parent SDQ subscales (0–10) ──────────────────────────────────────────
    psdq_cols = ["psdq_emotional", "psdq_peer", "psdq_conduct", "psdq_hyper", "psdq_prosocial"]
    for col in psdq_cols:
        if col in ("psdq_emotional", "psdq_peer"):
            vals = np.where(
                at_risk == 1,
                np.clip(rng.integers(4, 11, size=n), 0, 10),
                np.clip(rng.integers(0,  6, size=n), 0, 10),
            ).astype(float)
        elif col == "psdq_prosocial":
            vals = np.where(
                at_risk == 1,
                np.clip(rng.integers(3, 8, size=n), 0, 10),
                np.clip(rng.integers(6, 11, size=n), 0, 10),
            ).astype(float)
        else:
            vals = np.clip(rng.integers(0, 8, size=n), 0, 10).astype(float)
        rows[col] = _add_missing(vals, MISS_RATE, rng).tolist()

    psdq_arr = np.column_stack([
        np.nan_to_num([rows[c]], nan=0.0).flatten() for c in psdq_cols
    ])
    rows["psdq_total"] = psdq_arr.sum(axis=1).clip(0, 40).tolist()

    # ── Self SDQ subscales (0–10) ────────────────────────────────────────────
    sdq_cols = ["sdq_emotional", "sdq_peer", "sdq_conduct", "sdq_hyper", "sdq_prosocial"]
    for col in sdq_cols:
        if col in ("sdq_emotional", "sdq_peer"):
            vals = np.where(
                at_risk == 1,
                np.clip(rng.integers(4, 11, size=n), 0, 10),
                np.clip(rng.integers(0,  5, size=n), 0, 10),
            ).astype(float)
        elif col == "sdq_prosocial":
            vals = np.where(
                at_risk == 1,
                np.clip(rng.integers(3, 8, size=n), 0, 10),
                np.clip(rng.integers(6, 11, size=n), 0, 10),
            ).astype(float)
        else:
            vals = np.clip(rng.integers(0, 7, size=n), 0, 10).astype(float)
        rows[col] = _add_missing(vals, MISS_RATE, rng).tolist()

    sdq_arr = np.column_stack([
        np.nan_to_num([rows[c]], nan=0.0).flatten() for c in sdq_cols
    ])
    rows["sdq_total"] = sdq_arr.sum(axis=1).clip(0, 40).tolist()

    # ── WEMWBS wellbeing (14–70) ─────────────────────────────────────────────
    wemwbs = np.where(
        at_risk == 1,
        np.clip(rng.normal(38.0, 8.0, size=n), 14.0, 70.0),
        np.clip(rng.normal(53.0, 8.0, size=n), 14.0, 70.0),
    ).round(1)
    rows["wellbeing_wemwbs"] = _add_missing(wemwbs, MISS_RATE, rng).tolist()

    # ── Bullying flags (binary) ──────────────────────────────────────────────
    rows["bullied_flag"] = np.where(
        at_risk == 1,
        rng.choice([0, 1], p=[0.40, 0.60], size=n),
        rng.choice([0, 1], p=[0.70, 0.30], size=n),
    ).tolist()
    rows["cyberbullied_flag"] = rng.choice([0, 1], p=[0.75, 0.25], size=n).tolist()

    df = pd.DataFrame(rows)
    logger.info(
        "Generated %d rows | at-risk=%d (%.1f%%) | schools=%d",
        n, int(at_risk.sum()), 100 * at_risk.mean(), N_SCHOOLS,
    )
    return df


def main() -> None:
    cfg = Config()
    df = generate(cfg)

    # Write to the synthetic_v1 path (not the active real_v1)
    registry = cfg._dataset_registry  # noqa: SLF001
    syn_path = Path(registry["datasets"]["synthetic_v1"]["path"])
    syn_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(syn_path, index=False)
    logger.info("Saved → %s  (%d rows × %d cols)", syn_path, len(df), len(df.columns))


if __name__ == "__main__":
    main()
