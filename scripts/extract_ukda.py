"""
Extract UKDA-8467 (MHCYP 2017) into data/raw/real_v1.csv.

Dataset context
---------------
UKDA Study 8467 — Mental Health of Children and Young People in England 2017.
9,117 participants aged 2–19; DAWBA clinical assessments + SDQ + WEMWBS.
The dataset does NOT contain SPIN items. Clinical social phobia diagnoses
are derived from DAWBA structured interviews (ICD-10 / DSM-5).

Target variable
---------------
  dcsoph == 2  →  spin_risk_binary = 1  (ICD-10 Social Phobia, n=95, 1.04%)
  dcsoph == 0  →  spin_risk_binary = 0

ICD-10 (dcsoph, 95 cases) is preferred over DSM-5 (dcspph, 70 cases)
because ICD-10 criteria are less restrictive, giving more positive cases.

Feature groups extracted
------------------------
1. Identifiers      : participant_id, school_id (PSUa), cohort_wave
2. Metadata         : data_quality_flag, consent_obtained, spin_items_missing
3. Demographics     : age_years, gender, ethnicity_bme, deprivation_quintile, sen_status
4. Parent SDQ       : psdq_emotional … psdq_total  (all ages, parent-rated)
5. Self-report SDQ  : sdq_emotional … sdq_total    (11+ only; NaN for younger)
6. Wellbeing        : wellbeing_wemwbs              (11+ only; NaN for younger)
7. Bullying         : bullied_flag, cyberbullied_flag

PR item decision
----------------
PR0 (parent social concern gate) was investigated: 72/95 ICD-10 cases have
PR0=2 (no concern flagged), and PR10/PR5 use different scales entirely.
PR items are NOT reliable features for predicting clinical social phobia.
They are omitted from the output to keep the feature space clean.

Missing value treatment
-----------------------
SPSS user-missing codes (-1, -2 … -9) are replaced with NaN throughout.
Structural NaN in SDQ self-report / WEMWBS for ages <11 is expected —
the preprocessor handles these via MICE imputation.

Output
------
data/raw/real_v1.csv  (pipeline-ready; schema-compliant with schema.json v2.0)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

TAB_PATH = Path("/Users/mahesh/Documents/UKDA-8467-tab/tab/"
                "p11529_final_archiving_20210715_protect.tab")

OUT_PATH = Path("data/raw/real_v1.csv")

# ── SPSS missing codes ────────────────────────────────────────────────────────

_SPSS_MISSING = {-1, -2, -3, -4, -5, -6, -7, -8, -9}

# ── Columns to load from the raw tab file ────────────────────────────────────

_LOAD_COLS = [
    # Identifiers
    "seriala", "PSUa", "Year",
    # Target
    "dcsoph", "dcspph", "dcany",
    # Demographics
    "ChldAge", "ChldSx", "ethnic_bme", "imd_Quintiles",
    "ParentSN",     # Parent-reported SEN: 1=SEN, 2=no SEN (9.1% missing for 11-18)
    "Lrndifa",      # Learning difficulty level: 1=none, 2=mild, 3=moderate, 4=severe
    # Parent-rated SDQ (all ages, 5.6% missing). phyper absent — use phyper_corrected.
    "pemotion", "ppeer", "pconduct", "phyper_corrected", "pprosoc", "pebdtot",
    # Self-report SDQ (available for 11+; NaN for younger — not loaded here since we filter to 11+)
    "semotion", "speer", "sconduct", "shyper", "sprosoc", "sebdtot",
    # WEMWBS wellbeing (self-report, 11+)
    "wemwbs_sc",
    # Bullying: Bullyexp uses 1=bullied, 2=not bullied (NOT 0/1)
    "Bullyexp", "Cbullyexp",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _replace_spss_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace SPSS user-missing codes with NaN for numeric columns."""
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].where(~df[col].isin(_SPSS_MISSING))
    return df


def _to_int64(series: pd.Series, valid_min: float | None = None,
               valid_max: float | None = None) -> pd.Series:
    """Clip to valid range and cast to nullable Int64."""
    s = series.copy()
    if valid_min is not None:
        s = s.where(s >= valid_min)
    if valid_max is not None:
        s = s.where(s <= valid_max)
    return s.astype("Int64")


def _to_float64(series: pd.Series, valid_min: float | None = None,
                 valid_max: float | None = None) -> pd.Series:
    """Clip to valid range and cast to Float64."""
    s = series.copy()
    if valid_min is not None:
        s = s.where(s >= valid_min)
    if valid_max is not None:
        s = s.where(s <= valid_max)
    return s.astype("Float64")


# ── Transformation functions ──────────────────────────────────────────────────

def _recode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Derive spin_risk_binary from ICD-10 social phobia diagnosis."""
    # dcsoph: 0 = no diagnosis, 2 = ICD-10 social phobia diagnosis present
    df["spin_risk_binary"] = (df["dcsoph"] == 2).astype("Int64")
    return df


def _recode_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add pipeline metadata columns required by apply_exclusion_criteria()."""
    df["consent_obtained"] = 1          # all survey participants consented
    df["data_quality_flag"] = 0         # no quality exclusions applied here

    # Count missing parent SDQ items (replaces spin_items_missing in schema)
    psdq_cols = ["pemotion", "ppeer", "pconduct", "phyper", "pprosoc"]
    present = [c for c in psdq_cols if c in df.columns]
    df["spin_items_missing"] = df[present].isnull().sum(axis=1).astype("Int64")

    # Complete-case flag: 1 if all parent SDQ items present
    df["spin_complete_flag"] = (df["spin_items_missing"] == 0).astype("Int64")

    return df


def _recode_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Map UKDA demographic codes to pipeline values."""
    # Participant and group identifiers
    df["participant_id"] = df["seriala"].astype(str)
    df["school_id"] = df["PSUa"].astype(str)          # PSU cluster = school proxy
    df["cohort_wave"] = df["Year"].astype("Int64")

    # Age (after filtering to 11–18)
    df["age_years"] = _to_int64(df["ChldAge"], valid_min=11, valid_max=18)

    # Gender: 1=male → "male", 2=female → "female"
    df["gender"] = df["ChldSx"].map({1: "male", 2: "female"})

    # Ethnicity (binary BME flag; 0=white, 1=BME)
    df["ethnicity_bme"] = _to_int64(df["ethnic_bme"].map({0: 0, 1: 1}),
                                     valid_min=0, valid_max=1)

    # Deprivation quintile (1=least deprived, 5=most deprived)
    df["deprivation_quintile"] = _to_int64(df["imd_Quintiles"],
                                            valid_min=1, valid_max=5)

    # SEN status — ParentSN uses 1=SEN, 2=no SEN (NOT 0/1); recode to 0/1.
    # 9.1% missing for ages 11–18 (vs 92.6% for all ages).
    if "ParentSN" in df.columns:
        df["sen_status"] = _to_int64(df["ParentSN"].map({1: 1, 2: 0}),
                                      valid_min=0, valid_max=1)

    # Learning difficulty severity (1=none, 2=mild, 3=moderate, 4=severe).
    if "Lrndifa" in df.columns:
        df["learning_difficulty_level"] = _to_int64(df["Lrndifa"],
                                                     valid_min=1, valid_max=4)

    return df


def _recode_parent_sdq(df: pd.DataFrame) -> pd.DataFrame:
    """Clean parent-rated SDQ subscales.

    SDQ subscale scores are 0–10 (or 0–40 for total).
    Negative SPSS codes already replaced; here we clip to valid range.
    """
    mapping = {
        "pemotion":          ("psdq_emotional", 0, 10),
        "ppeer":             ("psdq_peer",      0, 10),
        "pconduct":          ("psdq_conduct",   0, 10),
        "phyper_corrected":  ("psdq_hyper",     0, 10),
        "pprosoc":           ("psdq_prosocial", 0, 10),
        "pebdtot":           ("psdq_total",     0, 40),
    }
    for src, (dst, lo, hi) in mapping.items():
        if src in df.columns:
            df[dst] = _to_int64(df[src], valid_min=lo, valid_max=hi)
    return df


def _recode_self_sdq(df: pd.DataFrame) -> pd.DataFrame:
    """Clean self-report SDQ subscales (collected only for age 11+).

    Participants younger than 11 will have NaN — handled by MICE in preprocessor.
    """
    mapping = {
        "semotion": ("sdq_emotional", 0, 10),
        "speer":    ("sdq_peer",      0, 10),
        "sconduct": ("sdq_conduct",   0, 10),
        "shyper":   ("sdq_hyper",     0, 10),
        "sprosoc":  ("sdq_prosocial", 0, 10),
        "sebdtot":  ("sdq_total",     0, 40),
    }
    for src, (dst, lo, hi) in mapping.items():
        if src in df.columns:
            df[dst] = _to_int64(df[src], valid_min=lo, valid_max=hi)
    return df


def _recode_wellbeing(df: pd.DataFrame) -> pd.DataFrame:
    """Clean WEMWBS total score (collected for age 11+).

    Valid range 14–70; negative codes already replaced with NaN.
    """
    if "wemwbs_sc" in df.columns:
        df["wellbeing_wemwbs"] = _to_float64(df["wemwbs_sc"],
                                              valid_min=14.0, valid_max=70.0)
    return df


def _recode_bullying(df: pd.DataFrame) -> pd.DataFrame:
    """Recode bullying experience flags to binary 0/1."""
    # Bullyexp uses 1=bullied, 2=not bullied (NOT 0/1 — confirmed from value counts)
    if "Bullyexp" in df.columns:
        df["bullied_flag"] = _to_int64(df["Bullyexp"].map({1: 1, 2: 0}),
                                        valid_min=0, valid_max=1)
    # Cbullyexp uses 0=not cyberbullied, 1=cyberbullied (standard 0/1)
    if "Cbullyexp" in df.columns:
        df["cyberbullied_flag"] = _to_int64(df["Cbullyexp"].map({0: 0, 1: 1}),
                                             valid_min=0, valid_max=1)
    return df


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the final output columns."""
    out_cols = [
        # Identifiers / metadata
        "participant_id", "school_id", "cohort_wave",
        "consent_obtained", "data_quality_flag",
        "spin_items_missing", "spin_complete_flag",
        # Target
        "spin_risk_binary",
        # Demographics
        "age_years", "gender", "ethnicity_bme", "deprivation_quintile",
        "sen_status", "learning_difficulty_level",
        # Parent-rated SDQ (all ages)
        "psdq_emotional", "psdq_peer", "psdq_conduct", "psdq_hyper",
        "psdq_prosocial", "psdq_total",
        # Self-report SDQ (11+ only)
        "sdq_emotional", "sdq_peer", "sdq_conduct", "sdq_hyper",
        "sdq_prosocial", "sdq_total",
        # Wellbeing (11+ only)
        "wellbeing_wemwbs",
        # Behavioural
        "bullied_flag", "cyberbullied_flag",
    ]
    return df[[c for c in out_cols if c in df.columns]]


# ── Quality report ────────────────────────────────────────────────────────────

def _print_quality_report(df: pd.DataFrame) -> None:
    n = len(df)
    n_pos = int(df["spin_risk_binary"].sum())
    logger.info("=" * 60)
    logger.info("UKDA-8467 Extraction Quality Report")
    logger.info("=" * 60)
    logger.info("Total rows         : %d", n)
    logger.info("Positive cases     : %d  (%.2f%%)", n_pos, 100 * n_pos / n)
    logger.info("Unique clusters    : %d", df["school_id"].nunique())
    logger.info("")
    logger.info("Age distribution (years):")
    age_counts = df["age_years"].value_counts().sort_index()
    for age, cnt in age_counts.items():
        logger.info("  Age %2d : %4d", age, cnt)
    logger.info("")
    logger.info("Missing rates on key features:")
    check_cols = [
        "psdq_emotional", "psdq_peer", "psdq_total",
        "sdq_emotional", "sdq_peer", "sdq_total",
        "wellbeing_wemwbs",
        "bullied_flag", "cyberbullied_flag",
        "deprivation_quintile", "sen_status", "learning_difficulty_level",
    ]
    for col in check_cols:
        if col in df.columns:
            miss = df[col].isna().sum()
            logger.info("  %-25s : %4d missing  (%.1f%%)", col, miss, 100 * miss / n)
    logger.info("")
    logger.info("Class balance:")
    logger.info("  Negative (no social phobia) : %d", n - n_pos)
    logger.info("  Positive (social phobia)    : %d", n_pos)
    logger.info("  Ratio                       : 1 : %.0f", (n - n_pos) / max(n_pos, 1))
    logger.info("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TAB_PATH.exists():
        raise FileNotFoundError(
            f"UKDA tab file not found at:\n  {TAB_PATH}\n"
            "Ensure the UKDA-8467 dataset is downloaded and the path is correct."
        )

    logger.info("Loading UKDA tab file (%s) ...", TAB_PATH.name)
    df = pd.read_csv(
        TAB_PATH,
        sep="\t",
        usecols=lambda c: c in _LOAD_COLS,
        low_memory=False,
    )
    logger.info("Loaded: %d rows × %d cols", *df.shape)

    # ── Step 1: Replace SPSS missing codes ───────────────────────────────────
    numeric_cols = [c for c in _LOAD_COLS if c not in ("seriala", "PSUa")]
    df = _replace_spss_missing(df, numeric_cols)

    # ── Step 1b: Filter to school age 11–18 ──────────────────────────────────
    # Restricts to the age band where self-report SDQ, WEMWBS, SEN (ParentSN),
    # and bullying (Bullyexp) are all collected (<16% missing vs 62–93% overall).
    age_col = pd.to_numeric(df["ChldAge"], errors="coerce")
    df = df[age_col.between(11, 18)].copy()
    df.reset_index(drop=True, inplace=True)
    logger.info("After age filter (11–18): %d rows", len(df))

    # ── Step 2: Derive all pipeline columns ───────────────────────────────────
    df = _recode_target(df)
    df = _recode_metadata(df)
    df = _recode_demographics(df)
    df = _recode_parent_sdq(df)
    df = _recode_self_sdq(df)
    df = _recode_wellbeing(df)
    df = _recode_bullying(df)

    # ── Step 3: Select final columns ─────────────────────────────────────────
    df = _select_output_columns(df)

    # ── Step 4: Print quality report ─────────────────────────────────────────
    _print_quality_report(df)

    # ── Step 5: Save ─────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    logger.info("Saved → %s  (%d rows, %d cols)", OUT_PATH, *df.shape)


if __name__ == "__main__":
    main()
