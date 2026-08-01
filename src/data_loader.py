"""
Data loader — resolves the active dataset from the registry and returns a
typed, schema-aligned DataFrame.

All file paths are read from ``config/dataset_registry.yaml`` via
``src.config.Config``; nothing is hard-coded here.

Usage
-----
>>> from src.config import Config
>>> from src.data_loader import load_active_dataset
>>> cfg = Config()
>>> df, meta = load_active_dataset(cfg)
>>> print(meta["label"], len(df))
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


# ── dtype map: built dynamically from schema so it stays in sync ─────────

def _build_dtype_map(cfg: Config) -> dict[str, str]:
    """Construct a pandas dtype map driven entirely by schema.json types."""
    type_map = {"string": "object", "integer": "Int64", "float": "Float64"}
    return {
        col["name"]: type_map.get(col["type"], "object")
        for col in cfg._columns  # noqa: SLF001 — intentional internal access
    }


def load_active_dataset(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Load the active dataset, apply schema dtypes, and return with metadata.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object.

    Returns
    -------
    df : pd.DataFrame
        Raw dataset with schema-aligned dtypes applied.
    meta : dict
        Active dataset metadata from ``dataset_registry.yaml``.

    Raises
    ------
    FileNotFoundError
        If the active dataset CSV does not exist on disk.
    """
    meta = cfg.active_dataset
    path = cfg.active_dataset_path

    if not path.exists():
        raise FileNotFoundError(
            f"Active dataset '{cfg.active_dataset_key}' not found at: {path}\n"
            "Run scripts/generate_synthetic_data.py to create synthetic data, "
            "or follow the transition checklist in dataset-mapping-strategy.md."
        )

    logger.info(
        "Loading dataset '%s' from %s", cfg.active_dataset_key, path
    )
    dtype_map = _build_dtype_map(cfg)

    df = pd.read_csv(
        path,
        dtype={k: v for k, v in dtype_map.items() if v != "Float64"},
        float_precision="high",
        na_values=["", "NA", "N/A", "nan", "NaN"],
        keep_default_na=True,
    )

    # Float64 (nullable) columns must be coerced separately
    float_cols = [
        col["name"]
        for col in cfg._columns  # noqa: SLF001
        if col["type"] == "float" and col["name"] in df.columns
    ]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")

    logger.info(
        "Loaded %d rows × %d columns (dataset: %s)",
        len(df),
        len(df.columns),
        cfg.active_dataset_key,
    )
    return df, meta


def load_dataset_by_key(cfg: Config, key: str) -> tuple[pd.DataFrame, dict]:
    """Load any registered dataset by its registry key.

    Useful for loading pilot or synthetic data alongside the active dataset
    for comparison or sensitivity analysis.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object.
    key : str
        Dataset key defined in ``dataset_registry.yaml``
        (e.g. ``'synthetic_v1'``, ``'real_v1'``).
    """
    original_active = cfg._dataset_registry["active"]  # noqa: SLF001
    cfg._dataset_registry["active"] = key  # noqa: SLF001
    try:
        df, meta = load_active_dataset(cfg)
    finally:
        cfg._dataset_registry["active"] = original_active  # noqa: SLF001
    return df, meta


def apply_exclusion_criteria(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Remove records that must not enter any analysis.

    Exclusion rules (from data-dictionary.md):
    1. ``consent_obtained != 1`` — no parental consent.
    2. ``data_quality_flag == 1`` — flagged as low quality.
    3. ``spin_items_missing > cfg.max_missing_spin_items`` — too many missing
       SPIN items to impute reliably.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    cfg : Config
        Configuration object.

    Returns
    -------
    pd.DataFrame
        Filtered dataset with excluded rows removed.
    """
    original_n = len(df)
    mask = pd.Series(True, index=df.index)

    # Rule 1 — consent (column may not be present in all dataset versions)
    if "consent_obtained" in df.columns:
        mask &= df["consent_obtained"].fillna(0).astype(int) == 1

    # Rule 2 — data quality flag
    if "data_quality_flag" in df.columns:
        mask &= df["data_quality_flag"].fillna(0).astype(int) != 1

    # Rule 3 — excessive SPIN/SDQ item missingness
    if "spin_items_missing" in df.columns:
        mask &= df["spin_items_missing"].fillna(0).astype(int) <= cfg.max_missing_spin_items
    else:
        # Compute on the fly; fall back to parent SDQ items for schema v2.0
        item_cols = cfg.spin_item_cols
        if not item_cols:
            sdq_subs = cfg._schema.get("sdq_subscales", {})
            item_cols = sdq_subs.get("parent_sdq_items", [])
        check_cols = [c for c in item_cols if c in df.columns]
        missing_counts = (
            df[check_cols].isna().sum(axis=1) if check_cols
            else pd.Series(0, index=df.index)
        )
        mask &= missing_counts <= cfg.max_missing_spin_items

    filtered = df[mask].copy().reset_index(drop=True)
    n_excluded = original_n - len(filtered)
    logger.info("Exclusion criteria applied: %d → %d rows (%d excluded)",
                original_n, len(filtered), n_excluded)
    return filtered
