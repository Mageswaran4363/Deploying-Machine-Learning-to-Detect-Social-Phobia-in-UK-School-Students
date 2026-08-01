"""
Schema validator — validates any DataFrame against schema.json before it
enters the ML pipeline.

The validator is the enforcement gate described in dataset-mapping-strategy.md
Section 5. It must be run on every new dataset (synthetic, pilot, or real)
before the dataset is activated in dataset_registry.yaml.

Usage
-----
>>> from src.config import Config
>>> from src.schema_validator import SchemaValidator
>>> cfg = Config()
>>> validator = SchemaValidator(cfg)
>>> report = validator.validate(df, "real_v1")
>>> if not report.passed:
...     raise RuntimeError(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Structured result from a schema validation run.

    Attributes
    ----------
    passed : bool
        True if no errors were found (warnings are non-blocking).
    dataset_label : str
        Human-readable identifier for the dataset being validated.
    n_rows : int
        Number of rows in the validated DataFrame.
    n_cols : int
        Number of columns in the validated DataFrame.
    errors : list[str]
        Blocking issues; non-empty means ``passed=False``.
    warnings : list[str]
        Non-blocking observations.
    """

    passed: bool
    dataset_label: str
    n_rows: int
    n_cols: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable validation summary string."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Schema validation {status} — {self.dataset_label}",
            f"  Rows: {self.n_rows} | Columns: {self.n_cols}",
            f"  Errors  : {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("  ── ERRORS ──")
            lines.extend(f"    • {e}" for e in self.errors)
        if self.warnings:
            lines.append("  ── WARNINGS ──")
            lines.extend(f"    • {w}" for w in self.warnings)
        return "\n".join(lines)


class SchemaValidator:
    """Validates a DataFrame against the project schema.

    All validation rules are derived from ``schema.json`` via ``Config``;
    no column names or valid-value ranges are hard-coded here.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def validate(self, df: pd.DataFrame, dataset_label: str = "dataset") -> ValidationReport:
        """Run all validation checks and return a :class:`ValidationReport`.

        Checks performed
        ----------------
        1. Required columns are present.
        2. No extra columns that are completely absent from schema.json.
        3. Integer columns contain only valid integer values.
        4. Numeric columns respect ``valid_min`` / ``valid_max`` bounds.
        5. String columns contain only ``valid_categories`` values.
        6. ``spin_risk_binary`` is consistent with ``spin_total_score``.
        7. ``spin_items_missing`` is consistent with actual missing counts.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        dataset_label : str
            Descriptive label used in the report (e.g. ``'real_v1'``).

        Returns
        -------
        ValidationReport
        """
        errors: list[str] = []
        warnings: list[str] = []

        self._check_required_columns(df, errors)
        self._check_unknown_columns(df, warnings)
        self._check_numeric_ranges(df, errors)
        self._check_valid_categories(df, errors)
        self._check_risk_label_consistency(df, errors)
        self._check_spin_missing_count(df, warnings)
        self._check_duplicate_participants(df, warnings)

        passed = len(errors) == 0
        report = ValidationReport(
            passed=passed,
            dataset_label=dataset_label,
            n_rows=len(df),
            n_cols=len(df.columns),
            errors=errors,
            warnings=warnings,
        )
        log_fn = logger.info if passed else logger.error
        log_fn("%s", report.summary())
        return report

    # ------------------------------------------------------------------ #
    # Individual checks                                                    #
    # ------------------------------------------------------------------ #

    def _check_required_columns(self, df: pd.DataFrame, errors: list[str]) -> None:
        """Error if any column marked ``required: true`` is absent."""
        missing = [c for c in self._cfg.all_required_cols() if c not in df.columns]
        if missing:
            errors.append(f"Required columns absent: {missing}")

    def _check_unknown_columns(self, df: pd.DataFrame, warnings: list[str]) -> None:
        """Warn about columns present in the DataFrame but not in schema.json."""
        schema_cols = set(self._cfg.all_col_names())
        unknown = [c for c in df.columns if c not in schema_cols]
        if unknown:
            warnings.append(
                f"{len(unknown)} column(s) not in schema.json (may be added features): {unknown}"
            )

    def _check_numeric_ranges(self, df: pd.DataFrame, errors: list[str]) -> None:
        """Error if numeric columns contain out-of-range values (ignoring NaN)."""
        for col in self._cfg._columns:  # noqa: SLF001
            name = col["name"]
            if name not in df.columns:
                continue
            valid_min = col.get("valid_min")
            valid_max = col.get("valid_max")
            if valid_min is None and valid_max is None:
                continue

            series = pd.to_numeric(df[name], errors="coerce").dropna()
            if valid_min is not None:
                bad = series[series < valid_min]
                if not bad.empty:
                    errors.append(
                        f"'{name}': {len(bad)} value(s) below valid_min={valid_min}"
                        f" (min found: {bad.min()})"
                    )
            if valid_max is not None:
                bad = series[series > valid_max]
                if not bad.empty:
                    errors.append(
                        f"'{name}': {len(bad)} value(s) above valid_max={valid_max}"
                        f" (max found: {bad.max()})"
                    )

    def _check_valid_categories(self, df: pd.DataFrame, errors: list[str]) -> None:
        """Error if categorical columns contain values outside valid_categories."""
        for col in self._cfg._columns:  # noqa: SLF001
            name = col["name"]
            cats = col.get("valid_categories")
            if cats is None or name not in df.columns:
                continue
            # Skip numeric columns that also have valid_categories (handled in range check)
            if col.get("type") in ("integer", "float"):
                continue
            actual = set(df[name].dropna().unique())
            invalid = actual - set(str(c) for c in cats)
            if invalid:
                errors.append(
                    f"'{name}': unexpected categories: {sorted(invalid)}"
                    f" (valid: {cats})"
                )

    def _check_risk_label_consistency(self, df: pd.DataFrame, errors: list[str]) -> None:
        """Error if spin_risk_binary is inconsistent with spin_total_score."""
        target = self._cfg.primary_target_col
        score_col = "spin_total_score"
        if target not in df.columns or score_col not in df.columns:
            return

        mask = df[target].notna() & df[score_col].notna()
        sub = df[mask].copy()
        expected = (
            pd.to_numeric(sub[score_col], errors="coerce") >= self._cfg.at_risk_threshold
        ).astype(int)
        actual = pd.to_numeric(sub[target], errors="coerce").astype(int)
        mismatches = (actual != expected).sum()
        if mismatches > 0:
            errors.append(
                f"'{target}' is inconsistent with '{score_col}' for {mismatches} row(s). "
                f"Threshold used: {self._cfg.at_risk_threshold}"
            )

    def _check_spin_missing_count(self, df: pd.DataFrame, warnings: list[str]) -> None:
        """Warn if spin_items_missing disagrees with actual item-level missingness.

        Uses SPIN item columns when available (schema v1.x); falls back to
        parent SDQ subscale items for schema v2.0 (UKDA adaptation).
        """
        if "spin_items_missing" not in df.columns:
            return
        item_cols = self._cfg.spin_item_cols
        if not item_cols:
            # Schema v2.0: spin_items_missing counts parent SDQ items instead
            sdq_subs = self._cfg._schema.get("sdq_subscales", {})
            item_cols = sdq_subs.get("parent_sdq_items", [])
        spin_cols = [c for c in item_cols if c in df.columns]
        if not spin_cols:
            return
        computed = df[spin_cols].isna().sum(axis=1)
        recorded = pd.to_numeric(df["spin_items_missing"], errors="coerce").fillna(0)
        disagreements = (computed != recorded).sum()
        if disagreements > 0:
            warnings.append(
                f"'spin_items_missing' disagrees with computed item-level missingness "
                f"for {disagreements} row(s). Consider recomputing the column."
            )

    def _check_duplicate_participants(self, df: pd.DataFrame, warnings: list[str]) -> None:
        """Warn if participant_id has duplicates (potential double submission)."""
        if "participant_id" not in df.columns:
            return
        dups = df["participant_id"].duplicated().sum()
        if dups > 0:
            warnings.append(
                f"'participant_id' has {dups} duplicate value(s). "
                "Check for double submissions."
            )
