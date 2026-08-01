"""
Feature engineering — stateless transformer that computes derived features
from schema-compliant raw variables.

Design principles
-----------------
* **Stateless** — ``transform()`` requires no ``fit()`` step; all operations
  are deterministic and reference only column names resolved from ``Config``.
* **Schema-driven** — SDQ subscale groupings are read from ``sdq_subscales``
  in schema.json v2.0. No column names are hard-coded.
* **Graceful degradation** — optional column groups are computed only when
  the relevant source columns are present; absent columns produce a logged
  debug message rather than an exception.

Usage
-----
>>> from src.config import Config
>>> from src.feature_engineering import FeatureEngineer
>>> cfg = Config()
>>> fe = FeatureEngineer(cfg)
>>> df_enriched = fe.transform(df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

_WEMWBS_LOW_THRESHOLD = 40.0  # approx. 1 SD below UK mean (51.9)


class FeatureEngineer:
    """Compute all engineered features from a schema-valid DataFrame.

    All source column names are resolved from ``cfg`` at instantiation.
    No column names are hard-coded in the logic below.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object (schema v2.0).
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        sdq_subs = cfg._schema.get("sdq_subscales", {})
        self._parent_internalising: list[str] = sdq_subs.get("parent_internalising", [])
        self._self_internalising: list[str]   = sdq_subs.get("self_internalising", [])
        self._parent_sdq_items: list[str]     = sdq_subs.get("parent_sdq_items", [])
        self._self_sdq_items: list[str]       = sdq_subs.get("self_sdq_items", [])

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all engineered features and append them to a copy of ``df``.

        Engineered feature groups
        -------------------------
        1. SDQ internalising composites (self + parent)
        2. Parent–self SDQ discrepancy
        3. Wellbeing threshold flag
        4. Composite bullying flag

        Parameters
        ----------
        df : pd.DataFrame
            Schema-valid DataFrame (post-imputation or raw).

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with engineered columns appended.
        """
        n_before = df.shape[1]
        df = df.copy()
        df = self._compute_sdq_internalising(df)
        df = self._compute_parent_self_discrepancy(df)
        df = self._compute_wellbeing_flag(df)
        df = self._compute_any_bullying(df)
        n_after = df.shape[1]
        logger.debug("FeatureEngineer added %d engineered column(s)", n_after - n_before)
        return df

    def engineered_col_names(self) -> list[str]:
        """Return the names of all columns this transformer adds."""
        return [
            "sdq_internalising",
            "psdq_internalising",
            "parent_self_discrepancy",
            "wellbeing_low_flag",
            "any_bullying",
        ]

    # ------------------------------------------------------------------ #
    # SDQ internalising composites                                         #
    # ------------------------------------------------------------------ #

    def _compute_sdq_internalising(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute internalising subscale composites (emotional + peer).

        sdq_internalising  = sdq_emotional + sdq_peer   (self-report, 0–20)
        psdq_internalising = psdq_emotional + psdq_peer (parent-rated, 0–20)

        The SDQ internalising score is the primary proxy for social anxiety
        in this dataset (highest discriminant validity for ICD-10 social phobia).
        """
        def _subscale_sum(cols: list[str], label: str) -> pd.Series:
            present = [c for c in cols if c in df.columns]
            absent  = [c for c in cols if c not in df.columns]
            if absent:
                logger.debug("%s: absent columns %s — excluded from sum", label, absent)
            if not present:
                return pd.Series(np.nan, index=df.index, dtype="Float64")
            return df[present].astype(float).sum(axis=1, skipna=False)

        if "sdq_internalising" not in df.columns:
            df["sdq_internalising"] = _subscale_sum(
                self._self_internalising, "sdq_internalising"
            )

        if "psdq_internalising" not in df.columns:
            df["psdq_internalising"] = _subscale_sum(
                self._parent_internalising, "psdq_internalising"
            )

        return df

    # ------------------------------------------------------------------ #
    # Parent–self discrepancy                                              #
    # ------------------------------------------------------------------ #

    def _compute_parent_self_discrepancy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the gap between parent-rated and self-rated SDQ total.

        parent_self_discrepancy = psdq_total − sdq_total

        Positive value: parent rates more difficulty than the child.
        Negative value: child rates more difficulty than the parent.

        Social phobia children often under-report (negative gap expected).
        Both source columns must be present; otherwise the feature is skipped.
        """
        if "parent_self_discrepancy" in df.columns:
            return df
        if "psdq_total" not in df.columns or "sdq_total" not in df.columns:
            logger.debug(
                "parent_self_discrepancy: 'psdq_total' or 'sdq_total' absent — skipped"
            )
            return df

        df["parent_self_discrepancy"] = (
            df["psdq_total"].astype(float) - df["sdq_total"].astype(float)
        )
        return df

    # ------------------------------------------------------------------ #
    # Wellbeing threshold flag                                             #
    # ------------------------------------------------------------------ #

    def _compute_wellbeing_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag participants below the WEMWBS clinical concern threshold.

        wellbeing_low_flag = 1 if wellbeing_wemwbs < 40.0 else 0

        WEMWBS ≥ 40 = acceptable wellbeing; < 40 indicates clinically
        meaningful low wellbeing (approx. 1 SD below UK mean of 51.9).
        NaN is preserved for participants with missing WEMWBS.
        """
        if "wellbeing_low_flag" in df.columns:
            return df
        if "wellbeing_wemwbs" not in df.columns:
            logger.debug("wellbeing_low_flag: 'wellbeing_wemwbs' absent — skipped")
            return df

        wemwbs = df["wellbeing_wemwbs"].astype(float)
        flag = pd.Series(np.where(wemwbs.isna(), np.nan, (wemwbs < _WEMWBS_LOW_THRESHOLD).astype(float)),
                          index=df.index)
        df["wellbeing_low_flag"] = flag
        return df

    # ------------------------------------------------------------------ #
    # Composite bullying flag                                              #
    # ------------------------------------------------------------------ #

    def _compute_any_bullying(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a composite bullying flag (school or cyber).

        any_bullying = 1 if bullied_flag == 1 OR cyberbullied_flag == 1

        If both source columns are present and non-null, any_bullying is
        deterministic.  If one is missing (NaN), the other drives the result
        where possible; otherwise NaN is propagated.
        """
        if "any_bullying" in df.columns:
            return df

        has_bullied = "bullied_flag" in df.columns
        has_cyber   = "cyberbullied_flag" in df.columns

        if not has_bullied and not has_cyber:
            logger.debug("any_bullying: no bullying columns present — skipped")
            return df

        if has_bullied and has_cyber:
            b = df["bullied_flag"].astype(float)
            c = df["cyberbullied_flag"].astype(float)
            df["any_bullying"] = np.where(
                b.isna() & c.isna(), np.nan,
                np.where(b == 1, 1.0, np.where(c == 1, 1.0, 0.0)),
            )
        elif has_bullied:
            df["any_bullying"] = df["bullied_flag"].astype(float)
        else:
            df["any_bullying"] = df["cyberbullied_flag"].astype(float)

        return df
