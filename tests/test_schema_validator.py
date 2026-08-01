"""Tests for src/schema_validator.py.

Coverage targets:
  SchemaValidator.validate()              — pass and fail paths
  _check_required_columns()              — absent required column
  _check_unknown_columns()               — extra column is a warning, not error
  _check_numeric_ranges()                — below valid_min, above valid_max
  _check_valid_categories()              — invalid string category
  _check_risk_label_consistency()        — spin_risk_binary vs spin_total_score
  _check_spin_missing_count()            — count disagreement
  _check_duplicate_participants()        — duplicate participant_id
  ValidationReport.summary()            — PASSED and FAILED formatting
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.schema_validator import SchemaValidator, ValidationReport


# ── Helpers ─────────────────────────────────────────────────────────────────

def _validator(cfg) -> SchemaValidator:
    return SchemaValidator(cfg)


# ── Happy-path tests ─────────────────────────────────────────────────────────

class TestValidDataPasses:
    def test_valid_df_passes(self, cfg, students_df):
        report = _validator(cfg).validate(students_df.copy(), "test")
        assert report.passed
        assert len(report.errors) == 0

    def test_passed_true_no_errors(self, cfg, students_df):
        report = _validator(cfg).validate(students_df.copy(), "test")
        assert report.passed is True

    def test_row_and_col_counts_reported(self, cfg, students_df):
        report = _validator(cfg).validate(students_df.copy(), "test")
        assert report.n_rows == len(students_df)
        assert report.n_cols == len(students_df.columns)

    def test_dataset_label_stored(self, cfg, students_df):
        report = _validator(cfg).validate(students_df.copy(), "my_dataset")
        assert report.dataset_label == "my_dataset"

    def test_extra_column_is_warning_not_error(self, cfg, students_df):
        df = students_df.copy()
        df["completely_unknown_xyz"] = 0
        report = _validator(cfg).validate(df, "test")
        assert report.passed
        assert any("completely_unknown_xyz" in w for w in report.warnings)

    def test_duplicate_participant_id_is_warning_not_error(self, cfg, students_df):
        df = students_df.copy()
        df.loc[0, "participant_id"] = df.loc[1, "participant_id"]
        report = _validator(cfg).validate(df, "test")
        assert report.passed  # duplicate is non-blocking
        assert any("duplicate" in w.lower() for w in report.warnings)

    def test_spin_items_missing_count_agreement_no_warning(self, cfg, students_df):
        # students_df has spin_items_missing=0 and no NaN in SPIN items
        report = _validator(cfg).validate(students_df.copy(), "test")
        spin_warnings = [w for w in report.warnings if "spin_items_missing" in w]
        assert len(spin_warnings) == 0

    def test_spin_items_missing_disagreement_is_warning(self, cfg, students_df):
        df = students_df.copy()
        # Claim more missing items than actually missing (actual = 0 NaN in parent SDQ)
        df["spin_items_missing"] = 5
        report = _validator(cfg).validate(df, "test")
        assert any("spin_items_missing" in w for w in report.warnings)

    def test_summary_contains_passed(self, cfg, students_df):
        report = _validator(cfg).validate(students_df.copy(), "test")
        assert "PASSED" in report.summary()

    def test_summary_contains_dataset_label(self, cfg, students_df):
        report = _validator(cfg).validate(students_df.copy(), "my_label")
        assert "my_label" in report.summary()


# ── Error-path tests ─────────────────────────────────────────────────────────

class TestRequiredColumnErrors:
    def test_missing_participant_id_causes_error(self, cfg, students_df):
        df = students_df.drop(columns=["participant_id"])
        report = _validator(cfg).validate(df, "test")
        assert not report.passed
        assert any("participant_id" in e for e in report.errors)

    def test_missing_school_id_causes_error(self, cfg, students_df):
        df = students_df.drop(columns=["school_id"])
        report = _validator(cfg).validate(df, "test")
        assert not report.passed
        assert any("school_id" in e for e in report.errors)

    def test_multiple_missing_columns_all_listed(self, cfg, students_df):
        df = students_df.drop(columns=["participant_id", "school_id"])
        report = _validator(cfg).validate(df, "test")
        assert not report.passed


class TestNumericRangeErrors:
    def test_sdq_item_above_max_causes_error(self, cfg, students_df):
        df = students_df.copy()
        df.loc[0, "psdq_emotional"] = 11  # valid_max = 10
        report = _validator(cfg).validate(df, "test")
        assert not report.passed
        assert any("psdq_emotional" in e for e in report.errors)

    def test_sdq_item_below_min_causes_error(self, cfg, students_df):
        df = students_df.copy()
        df.loc[0, "psdq_emotional"] = -1  # valid_min = 0
        report = _validator(cfg).validate(df, "test")
        assert not report.passed
        assert any("psdq_emotional" in e for e in report.errors)

    def test_age_above_max_causes_error(self, cfg, students_df):
        df = students_df.copy()
        df.loc[0, "age_years"] = 99  # valid_max = 18
        report = _validator(cfg).validate(df, "test")
        assert not report.passed

    def test_nan_values_ignored_by_range_check(self, cfg, students_df):
        df = students_df.copy()
        df.loc[0, "age_years"] = float("nan")  # NaN should not trigger a range error
        report = _validator(cfg).validate(df, "test")
        assert report.passed


class TestCategoryErrors:
    def test_invalid_gender_value_causes_error(self, cfg, students_df):
        df = students_df.copy()
        df.loc[0, "gender"] = "other"   # not in valid_categories ["male", "female"]
        report = _validator(cfg).validate(df, "test")
        assert not report.passed
        assert any("gender" in e for e in report.errors)

    def test_invalid_gender_causes_error(self, cfg, students_df):
        df = students_df.copy()
        df.loc[0, "gender"] = "attack_helicopter"
        report = _validator(cfg).validate(df, "test")
        assert not report.passed

    def test_valid_gender_categories_no_error(self, cfg, students_df):
        # "male" and "female" are valid — no error expected
        report = _validator(cfg).validate(students_df.copy(), "test")
        category_errors = [e for e in report.errors if "gender" in e]
        assert len(category_errors) == 0


class TestRiskLabelConsistency:
    def test_consistent_labels_no_error(self, cfg, students_df):
        # students_df has spin_risk_binary derived from spin_total_score
        report = _validator(cfg).validate(students_df.copy(), "test")
        consistency_errors = [e for e in report.errors if "inconsistent" in e.lower()]
        assert len(consistency_errors) == 0

    def test_inconsistent_label_causes_error(self, cfg, students_df):
        df = students_df.copy()
        if "spin_total_score" in df.columns:
            # Force an inconsistency: set total_score high but mark as 0
            df.loc[0, "spin_total_score"] = 30
            df.loc[0, "spin_risk_binary"] = 0
            report = _validator(cfg).validate(df, "test")
            assert not report.passed

    def test_check_skipped_when_total_score_absent(self, cfg, students_df):
        # Without spin_total_score, the consistency check is skipped entirely
        df = students_df.drop(columns=["spin_total_score"], errors="ignore")
        df.loc[0, "spin_risk_binary"] = 99  # would fail if check ran
        report = _validator(cfg).validate(df, "test")
        # Only numeric range errors, no consistency error
        consistency_errors = [e for e in report.errors if "inconsistent" in e.lower()]
        assert len(consistency_errors) == 0


class TestValidationReportSummary:
    def test_summary_contains_failed_on_error(self, cfg, students_df):
        df = students_df.drop(columns=["participant_id"])
        report = _validator(cfg).validate(df, "test")
        assert "FAILED" in report.summary()

    def test_summary_error_section_lists_errors(self, cfg, students_df):
        df = students_df.drop(columns=["participant_id"])
        report = _validator(cfg).validate(df, "test")
        summary = report.summary()
        assert "ERRORS" in summary or "participant_id" in summary

    def test_summary_warning_section_present(self, cfg, students_df):
        df = students_df.copy()
        df["extra_col_xyz"] = 0
        report = _validator(cfg).validate(df, "test")
        assert "WARNINGS" in report.summary() or "extra_col_xyz" in report.summary()

    def test_passed_false_when_errors_present(self, cfg, students_df):
        df = students_df.drop(columns=["participant_id"])
        report = _validator(cfg).validate(df, "test")
        assert report.passed is False
