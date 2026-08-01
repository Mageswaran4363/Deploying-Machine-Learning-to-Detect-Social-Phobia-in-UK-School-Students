"""
Configuration module — single source of truth for all pipeline constants.

Loads ``config/dataset_registry.yaml``, ``config/model_registry.yaml``, and
``config/schema.json`` and exposes derived column lists so that no other
module ever hard-codes a column name.

Usage
-----
>>> from src.config import Config
>>> cfg = Config()
>>> cfg.spin_item_cols
['spin_q01', 'spin_q02', ..., 'spin_q17']
>>> cfg.primary_target_col
'spin_risk_binary'
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"


class Config:
    """Central configuration object for the ML pipeline.

    All column names, file paths, and hyperparameter defaults are resolved
    from the config files — never hard-coded in pipeline logic.

    Parameters
    ----------
    config_dir : Path, optional
        Directory containing ``dataset_registry.yaml``,
        ``model_registry.yaml``, and ``schema.json``.
        Defaults to ``<project_root>/config``.
    """

    def __init__(self, config_dir: Path = _DEFAULT_CONFIG_DIR) -> None:
        self._config_dir = Path(config_dir)
        self._dataset_registry: dict = self._load_yaml("dataset_registry.yaml")
        self._model_registry: dict = self._load_yaml("model_registry.yaml")
        self._schema: dict = self._load_json("schema.json")
        self._columns: list[dict] = self._schema["columns"]
        logger.info(
            "Config loaded | schema v%s | active dataset: %s",
            self._schema.get("version", "?"),
            self._dataset_registry.get("active", "?"),
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_yaml(self, filename: str) -> dict:
        path = self._config_dir / filename
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _load_json(self, filename: str) -> dict:
        path = self._config_dir / filename
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def _cols_where(self, **filters: Any) -> list[str]:
        """Return column names whose schema entry matches all key=value filters."""
        return [
            col["name"]
            for col in self._columns
            if all(col.get(k) == v for k, v in filters.items())
        ]

    def _col_meta_where(self, **filters: Any) -> list[dict]:
        """Return full schema dicts for columns matching all key=value filters."""
        return [
            col
            for col in self._columns
            if all(col.get(k) == v for k, v in filters.items())
        ]

    # ------------------------------------------------------------------ #
    # Dataset registry                                                     #
    # ------------------------------------------------------------------ #

    @property
    def active_dataset_key(self) -> str:
        """Active dataset version key (e.g. ``'real_v1'``)."""
        return self._dataset_registry["active"]

    @property
    def active_dataset(self) -> dict:
        """Full metadata dict for the active dataset."""
        return self._dataset_registry["datasets"][self.active_dataset_key]

    @property
    def active_dataset_path(self) -> Path:
        """Resolved filesystem path to the active dataset CSV."""
        return Path(self.active_dataset["path"])

    def get_dataset_meta(self, key: str) -> dict:
        """Return the metadata dict for a specific dataset version key."""
        return self._dataset_registry["datasets"][key]

    # ------------------------------------------------------------------ #
    # Model registry                                                       #
    # ------------------------------------------------------------------ #

    @property
    def model_defaults(self) -> dict:
        """Training pipeline defaults from model_registry.yaml."""
        return self._model_registry.get("model_defaults", {})

    @property
    def random_seed(self) -> int:
        return int(self.model_defaults.get("random_seed", 42))

    @property
    def cv_outer_folds(self) -> int:
        return int(self.model_defaults.get("cv_outer_folds", 5))

    @property
    def cv_inner_folds(self) -> int:
        return int(self.model_defaults.get("cv_inner_folds", 3))

    @property
    def optuna_n_trials(self) -> int:
        return int(self.model_defaults.get("optuna_n_trials", 50))

    @property
    def smote_sampling_strategy(self) -> float:
        return float(self.model_defaults.get("smote_sampling_strategy", 0.30))

    @property
    def smote_k_neighbors(self) -> int:
        return int(self.model_defaults.get("smote_k_neighbors", 5))

    @property
    def recall_minimum(self) -> float:
        """Hard deployment gate: model is not deployed if Recall < this."""
        return float(self.model_defaults.get("recall_minimum", 0.85))

    @property
    def auc_roc_minimum(self) -> float:
        """Hard deployment gate: model is not deployed if AUC-ROC < this."""
        return float(self.model_defaults.get("auc_roc_minimum", 0.80))

    @property
    def pr_auc_minimum(self) -> float:
        return float(self.model_defaults.get("pr_auc_minimum", 0.50))

    @property
    def selection_weights(self) -> dict[str, float]:
        """Weights used to compute the composite model selection score."""
        return self.model_defaults.get(
            "selection_weights",
            {"recall": 0.45, "pr_auc": 0.25, "f1": 0.20, "auc_roc": 0.10},
        )

    @property
    def interpretability_rank(self) -> list[str]:
        """Model names ordered from most to least interpretable (tie-break)."""
        return self.model_defaults.get(
            "interpretability_rank",
            ["DecisionTree", "LogisticRegression", "RandomForest", "XGBoost", "SVM", "MLP"],
        )

    # ------------------------------------------------------------------ #
    # Schema — top-level properties                                        #
    # ------------------------------------------------------------------ #

    @property
    def schema_version(self) -> str:
        return self._schema.get("version", "unknown")

    @property
    def group_col(self) -> str:
        """Column used for group stratification in CV (school identifier)."""
        return self._schema.get("group_col", "school_id")

    @property
    def max_missing_spin_items(self) -> int:
        """Records exceeding this missing-item count are excluded."""
        return int(self._schema.get("max_missing_spin_items", 5))

    @property
    def split_ratios(self) -> dict[str, float]:
        return self._schema.get(
            "split_ratios", {"train": 0.70, "validation": 0.15, "test": 0.15}
        )

    # ------------------------------------------------------------------ #
    # Risk cut-offs                                                        #
    # ------------------------------------------------------------------ #

    @property
    def risk_cutoffs(self) -> dict:
        return self._schema["risk_cutoffs"]

    @property
    def at_risk_threshold(self) -> int:
        """SPIN total score at or above which a student is classified at-risk."""
        return int(self.risk_cutoffs["at_risk_threshold"])

    @property
    def moderate_lower(self) -> int:
        return int(self.risk_cutoffs["moderate_lower"])

    @property
    def high_lower(self) -> int:
        return int(self.risk_cutoffs["high_lower"])

    # ------------------------------------------------------------------ #
    # SPIN subscales                                                       #
    # ------------------------------------------------------------------ #

    @property
    def spin_subscales(self) -> dict[str, list[str]]:
        """Mapping of subscale name to list of item column names."""
        return self._schema["spin_subscales"]

    @property
    def fear_items(self) -> list[str]:
        return self.spin_subscales["fear"]

    @property
    def avoidance_items(self) -> list[str]:
        return self.spin_subscales["avoidance"]

    @property
    def physiological_items(self) -> list[str]:
        return self.spin_subscales["physiological"]

    @property
    def spin_max_scores(self) -> dict[str, int]:
        return self._schema["spin_max_scores"]

    # ------------------------------------------------------------------ #
    # Column groups (derived from schema)                                  #
    # ------------------------------------------------------------------ #

    @property
    def spin_item_cols(self) -> list[str]:
        return self._cols_where(group="spin_item")

    @property
    def spin_derived_cols(self) -> list[str]:
        return self._cols_where(group="spin_derived")

    @property
    def demographic_cols(self) -> list[str]:
        return self._cols_where(group="demographic")

    @property
    def attendance_cols(self) -> list[str]:
        return self._cols_where(group="attendance")

    @property
    def academic_cols(self) -> list[str]:
        return self._cols_where(group="academic")

    @property
    def identifier_cols(self) -> list[str]:
        return self._cols_where(group="identifier")

    @property
    def metadata_cols(self) -> list[str]:
        return self._cols_where(group="metadata")

    # ------------------------------------------------------------------ #
    # Modelling column sets                                                #
    # ------------------------------------------------------------------ #

    @property
    def model_feature_cols(self) -> list[str]:
        """All columns used as input features in the ML model."""
        return self._cols_where(used_in_modelling=True)

    @property
    def primary_target_col(self) -> str:
        """The single primary binary classification target column."""
        cols = self._cols_where(is_primary_target=True)
        if len(cols) != 1:
            raise ValueError(
                f"Expected exactly one primary target; found {len(cols)}: {cols}"
            )
        return cols[0]

    @property
    def fairness_audit_cols(self) -> list[str]:
        """Columns used to stratify the fairness / bias audit."""
        return self._cols_where(used_in_fairness_audit=True)

    # ------------------------------------------------------------------ #
    # Encoding groups                                                      #
    # ------------------------------------------------------------------ #

    @property
    def ordinal_cols(self) -> list[str]:
        """Modelling columns to be ordinally encoded."""
        return [
            c["name"]
            for c in self._columns
            if c.get("encoding") == "ordinal" and c.get("used_in_modelling")
        ]

    @property
    def onehot_cols(self) -> list[str]:
        """Modelling columns to be one-hot encoded."""
        return [
            c["name"]
            for c in self._columns
            if c.get("encoding") == "onehot" and c.get("used_in_modelling")
        ]

    @property
    def passthrough_numeric_cols(self) -> list[str]:
        """Numeric modelling columns passed through without encoding."""
        return [
            c["name"]
            for c in self._columns
            if c.get("encoding") == "passthrough"
            and c.get("used_in_modelling")
            and c.get("type") in ("integer", "float")
        ]

    @property
    def binary_cols(self) -> list[str]:
        """Binary (0/1) modelling columns (no encoding needed)."""
        return [
            c["name"]
            for c in self._columns
            if c.get("encoding") == "binary" and c.get("used_in_modelling")
        ]

    @property
    def log_standardise_cols(self) -> list[str]:
        """Columns requiring log1p transform before standardisation."""
        return [
            c["name"]
            for c in self._columns
            if c.get("encoding") == "log_standardise" and c.get("used_in_modelling")
        ]

    @property
    def scalable_cols(self) -> list[str]:
        """All modelling columns that should be standardised for linear models."""
        return [
            c["name"]
            for c in self._columns
            if c.get("apply_scaling") and c.get("used_in_modelling")
        ]

    # ------------------------------------------------------------------ #
    # Imputation groups                                                    #
    # ------------------------------------------------------------------ #

    @property
    def mice_cols(self) -> list[str]:
        """Columns imputed via MICE (IterativeImputer)."""
        return self._cols_where(imputation_strategy="mice")

    @property
    def knn_cols(self) -> list[str]:
        """Columns imputed via KNNImputer."""
        return self._cols_where(imputation_strategy="knn")

    @property
    def zero_impute_cols(self) -> list[str]:
        """Columns where missing values are filled with 0."""
        return self._cols_where(imputation_strategy="zero")

    @property
    def mode_impute_cols(self) -> list[str]:
        """Columns where missing values are filled with the mode."""
        return self._cols_where(imputation_strategy="mode")

    @property
    def median_impute_cols(self) -> list[str]:
        """Columns where missing values are filled with the median."""
        return self._cols_where(imputation_strategy="median")

    @property
    def conservative_min_cols(self) -> list[str]:
        """MNAR-aware columns: missing → minimum valid value (pessimistic imputation)."""
        return self._cols_where(imputation_strategy="conservative_min")

    # ------------------------------------------------------------------ #
    # Per-column metadata helpers                                          #
    # ------------------------------------------------------------------ #

    def get_column_meta(self, col_name: str) -> dict:
        """Return the full schema metadata dict for a single column.

        Raises
        ------
        KeyError
            If ``col_name`` is not defined in schema.json.
        """
        for col in self._columns:
            if col["name"] == col_name:
                return col
        raise KeyError(f"Column '{col_name}' not found in schema.json")

    def get_lay_label(self, col_name: str) -> str:
        """Human-readable label for SHAP output; falls back to column name."""
        meta = self.get_column_meta(col_name)
        return meta.get("lay_label") or col_name

    def get_ordinal_categories(self, col_name: str) -> list:
        """Ordered category list for an ordinal column.

        Raises
        ------
        KeyError
            If the column has no ``ordinal_categories`` defined.
        """
        meta = self.get_column_meta(col_name)
        cats = meta.get("ordinal_categories")
        if cats is None:
            raise KeyError(
                f"Column '{col_name}' has no 'ordinal_categories' in schema.json"
            )
        return cats

    def get_valid_categories(self, col_name: str) -> list | None:
        """Valid string/integer category values for a column, or None."""
        meta = self.get_column_meta(col_name)
        return meta.get("valid_categories")

    def get_valid_range(self, col_name: str) -> tuple[float | None, float | None]:
        """Return (valid_min, valid_max) for numeric columns."""
        meta = self.get_column_meta(col_name)
        return meta.get("valid_min"), meta.get("valid_max")

    def is_column_required(self, col_name: str) -> bool:
        """True if the column must be present in every dataset."""
        meta = self.get_column_meta(col_name)
        return bool(meta.get("required", False))

    def all_required_cols(self) -> list[str]:
        """All columns marked ``required: true`` in schema.json."""
        return self._cols_where(required=True)

    def all_col_names(self) -> list[str]:
        """Complete list of column names defined in schema.json."""
        return [c["name"] for c in self._columns]
