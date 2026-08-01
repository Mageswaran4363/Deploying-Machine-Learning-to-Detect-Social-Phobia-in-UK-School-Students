"""
Preprocessing pipeline — imputation, encoding, scaling, and train/test split.

Key design rules (from ml-development-plan.md Section 2):
* All imputers are **fit on training data only**; transform is applied to
  validation and test sets using the already-fitted imputer.
* SMOTE is applied to the training fold only (never validation or test).
* Column lists for each step are resolved from ``Config`` (schema-driven).
* The fitted ``Preprocessor`` object is serialised to disk for reproducibility
  and for use at inference time in the web application.

Usage
-----
>>> from src.config import Config
>>> from src.preprocessing import Preprocessor, split_dataset
>>> cfg = Config()
>>> X_train, X_val, X_test, y_train, y_val, y_test, groups_train = split_dataset(df, cfg)
>>> prep = Preprocessor(cfg)
>>> X_train_proc = prep.fit_transform(X_train, groups=groups_train)
>>> X_val_proc   = prep.transform(X_val)
>>> X_test_proc  = prep.transform(X_test)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.config import Config

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────

def _present(cols: list[str], df: pd.DataFrame) -> list[str]:
    """Filter a column list to those actually present in the DataFrame."""
    return [c for c in cols if c in df.columns]


# ── Train / test split ────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the dataset into train, validation, and held-out test sets.

    Strategy (from ml-development-plan.md Section 4):
    - Held-out test set is composed of a held-out school (group-isolated).
    - Remaining data is split 70/15 into train/validation with
      ``StratifiedGroupKFold`` by school and risk label.

    Parameters
    ----------
    df : pd.DataFrame
        Full, exclusion-filtered dataset.
    cfg : Config
        Configuration object.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_val, y_test : pd.Series
        Target vectors (spin_risk_binary).
    groups_train : pd.Series
        School IDs for the training set (used in nested CV).
    """
    warnings.warn(
        "split_dataset() is not used by the nested CV pipeline and will be removed "
        "in a future version. Use _run_nested_cv() via run_training_pipeline() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    target = cfg.primary_target_col
    group_col = cfg.group_col

    # Separate features and target
    y = df[target].astype(int)
    groups = df[group_col]

    # Feature matrix: keep only modelling columns + group col for splitting
    feature_cols = cfg.model_feature_cols
    available_features = _present(feature_cols, df)
    X = df[available_features].copy()

    # Split off a single school as the final held-out test set
    unique_schools = groups.unique()
    n_schools = len(unique_schools)

    # Adapt fold count to available schools (can't have more folds than groups)
    n_outer = min(cfg.cv_outer_folds, n_schools)
    if n_outer < cfg.cv_outer_folds:
        logger.warning(
            "Only %d school(s) available; reducing outer CV folds from %d to %d",
            n_schools, cfg.cv_outer_folds, n_outer,
        )

    sgkf = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=cfg.random_seed)

    # Use the first fold's test indices as the held-out test set
    splits = list(sgkf.split(X, y, groups=groups))
    train_val_idx, test_idx = splits[0]

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    # Further split train_val into train / validation (second fold)
    n_inner = min(cfg.cv_inner_folds, groups_train_val.nunique())
    inner_sgkf = StratifiedGroupKFold(
        n_splits=n_inner, shuffle=True, random_state=cfg.random_seed
    )
    inner_splits = list(inner_sgkf.split(X_train_val, y_train_val, groups=groups_train_val))
    train_idx_inner, val_idx_inner = inner_splits[0]

    X_train = X_train_val.iloc[train_idx_inner].reset_index(drop=True)
    y_train = y_train_val.iloc[train_idx_inner].reset_index(drop=True)
    groups_train = groups_train_val.iloc[train_idx_inner].reset_index(drop=True)

    X_val = X_train_val.iloc[val_idx_inner].reset_index(drop=True)
    y_val = y_train_val.iloc[val_idx_inner].reset_index(drop=True)

    logger.info(
        "Dataset split → train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    logger.info(
        "Class distribution — train: %.1f%% at-risk | val: %.1f%% | test: %.1f%%",
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train


# ── Preprocessor class ────────────────────────────────────────────────────

class Preprocessor:
    """Fit-transform pipeline for imputation, encoding, and scaling.

    All column references are resolved from ``Config`` at construction time.
    No column names are hard-coded in the logic.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._fitted = False

        # Imputers — initialised but not fitted
        # min_value/max_value are set at fit time from schema column ranges.
        self._spin_imputer = IterativeImputer(
            max_iter=10,
            random_state=cfg.random_seed,
            initial_strategy="median",
        )
        self._fitted_spin_clip_ranges: list[tuple[float, float]] = []
        self._knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
        self._median_imputer = SimpleImputer(strategy="median")
        self._mode_imputer = SimpleImputer(strategy="most_frequent")

        # Ordinal encoder — fitted columns determined at fit time
        self._ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=np.nan,
        )

        # Scaler — applied to continuous features for linear/SVM/MLP models
        self._scaler = StandardScaler()

        # Column lists resolved by schema — filtered to those actually present
        # at fit time (stored in fit_transform)
        self._fitted_spin_cols: list[str] = []
        self._fitted_knn_cols: list[str] = []
        self._fitted_median_cols: list[str] = []
        self._fitted_mode_cols: list[str] = []
        self._fitted_ordinal_cols: list[str] = []
        self._fitted_ordinal_categories: list[list] = []
        self._fitted_scale_cols: list[str] = []
        self._feature_names_out: list[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit_transform(self, X: pd.DataFrame, groups: pd.Series | None = None) -> pd.DataFrame:
        """Fit all transformers on X and return the transformed DataFrame.

        Must be called before ``transform()``.  All imputers, encoders, and
        the scaler are fitted here and stored for use on validation/test sets.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix (post feature-engineering, pre-imputation).
        groups : pd.Series, optional
            School IDs (unused here; accepted for API symmetry with CV loops).

        Returns
        -------
        pd.DataFrame
            Transformed training features.
        """
        if self._fitted:
            logger.warning(
                "Preprocessor.fit_transform() called on an already-fitted object. "
                "Create a new Preprocessor instance for a fresh fit."
            )

        X = X.copy()

        # 1 — Apply conservative_min imputation (MNAR; before MICE)
        X = self._apply_conservative_min(X)

        # 2 — Zero-fill absence / SEN columns
        X = self._apply_zero_impute(X)

        # 3 — MICE for SPIN/SDQ/WEMWBS columns (fit here)
        spin_cols = _present(self._cfg.mice_cols, X)
        if spin_cols:
            self._fitted_spin_cols = spin_cols
            min_vals = [self._cfg.get_valid_range(c)[0] or 0.0 for c in spin_cols]
            max_vals = [self._cfg.get_valid_range(c)[1] or 1e6 for c in spin_cols]
            self._spin_imputer.set_params(min_value=min_vals, max_value=max_vals)
            self._fitted_spin_clip_ranges = list(zip(min_vals, max_vals))
            X[spin_cols] = self._spin_imputer.fit_transform(X[spin_cols].astype(float))
            for col, (lo, hi) in zip(spin_cols, self._fitted_spin_clip_ranges):
                X[col] = X[col].clip(lo, hi)

        # 4 — KNN imputation for academic variables (fit here)
        knn_cols = _present(self._cfg.knn_cols, X)
        if knn_cols:
            self._fitted_knn_cols = knn_cols
            X[knn_cols] = self._knn_imputer.fit_transform(X[knn_cols].astype(float))

        # 5 — Median imputation for age, attendance (fit here)
        med_cols = _present(self._cfg.median_impute_cols, X)
        if med_cols:
            self._fitted_median_cols = med_cols
            X[med_cols] = self._median_imputer.fit_transform(X[med_cols].astype(float))

        # 6 — Mode imputation (fit here)
        mode_cols = _present(self._cfg.mode_impute_cols, X)
        if mode_cols:
            self._fitted_mode_cols = mode_cols
            X[mode_cols] = self._mode_imputer.fit_transform(X[mode_cols])

        # 6b — Recompute SDQ derived features from now-imputed source columns.
        # Must run after MICE (SDQ/WEMWBS) and mode (bullying flags) are filled.
        X = self._recompute_sdq_derived(X)

        # 7 — Ordinal encoding (fit here)
        X = self._fit_ordinal_encoder(X)

        # 8 — Standard scaling (fit here)
        scale_cols = _present(self._cfg.scalable_cols, X)
        # Only scale columns that actually exist and are numeric after encoding
        scale_cols = [c for c in scale_cols if c in X.columns and X[c].dtype != object]
        if scale_cols:
            self._fitted_scale_cols = scale_cols
            X[scale_cols] = self._scaler.fit_transform(X[scale_cols].astype(float))

        self._feature_names_out = list(X.columns)
        self._fitted = True
        logger.info(
            "Preprocessor fitted on %d rows × %d features", len(X), len(X.columns)
        )
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformers to validation or test data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (validation or test split).

        Returns
        -------
        pd.DataFrame
            Transformed features with the same column layout as fit_transform output.

        Raises
        ------
        RuntimeError
            If called before ``fit_transform()``.
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor.transform() called before fit_transform(). "
                "Call fit_transform() on training data first."
            )
        X = X.copy()

        X = self._apply_conservative_min(X)
        X = self._apply_zero_impute(X)

        if self._fitted_spin_cols:
            present = _present(self._fitted_spin_cols, X)
            X[present] = self._spin_imputer.transform(X[present].astype(float))
            clip_map = dict(zip(self._fitted_spin_cols, self._fitted_spin_clip_ranges))
            for col in present:
                lo, hi = clip_map[col]
                X[col] = X[col].clip(lo, hi)

        if self._fitted_knn_cols:
            present = _present(self._fitted_knn_cols, X)
            X[present] = self._knn_imputer.transform(X[present].astype(float))

        if self._fitted_median_cols:
            present = _present(self._fitted_median_cols, X)
            X[present] = self._median_imputer.transform(X[present].astype(float))

        if self._fitted_mode_cols:
            present = _present(self._fitted_mode_cols, X)
            X[present] = self._mode_imputer.transform(X[present])

        X = self._recompute_sdq_derived(X)

        X = self._apply_ordinal_encoder(X)

        if self._fitted_scale_cols:
            present = _present(self._fitted_scale_cols, X)
            X[present] = self._scaler.transform(X[present].astype(float))

        # Align columns to training schema (add missing cols as NaN, drop extras)
        X = self._align_columns(X)
        return X

    @property
    def feature_names_out(self) -> list[str]:
        """Column names of the output DataFrame after fit_transform."""
        if not self._fitted:
            raise RuntimeError("fit_transform() must be called before accessing feature_names_out")
        return self._feature_names_out

    def save(self, path: str | Path) -> None:
        """Serialise the fitted Preprocessor to disk.

        Parameters
        ----------
        path : str or Path
            Target file path (e.g. ``models/preprocessor_v1.pkl``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        """Load a previously saved Preprocessor from disk.

        Parameters
        ----------
        path : str or Path
            Path to a ``.pkl`` file created by ``save()``.
        """
        path = Path(path)
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected Preprocessor, got {type(obj)}")
        logger.info("Preprocessor loaded from %s", path)
        return obj

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _apply_conservative_min(self, X: pd.DataFrame) -> pd.DataFrame:
        """MNAR-aware imputation: fill with the minimum valid value.

        Applied to participation/engagement ratings where a missing teacher
        rating most likely indicates a student who never participates.
        """
        for col in _present(self._cfg.conservative_min_cols, X):
            valid_min, _ = self._cfg.get_valid_range(col)
            fill_val = valid_min if valid_min is not None else 1
            X[col] = X[col].fillna(fill_val)
        return X

    def _apply_zero_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill absence counts and SEN flags with 0 when missing."""
        for col in _present(self._cfg.zero_impute_cols, X):
            X[col] = X[col].fillna(0)
        return X

    def _fit_ordinal_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit OrdinalEncoder on ordinal columns and transform in place."""
        ord_cols = _present(self._cfg.ordinal_cols, X)
        if not ord_cols:
            return X
        self._fitted_ordinal_cols = ord_cols
        # Build per-column category lists from schema
        categories = [self._cfg.get_ordinal_categories(c) for c in ord_cols]
        self._fitted_ordinal_categories = categories
        self._ordinal_encoder.set_params(categories=categories)
        X[ord_cols] = self._ordinal_encoder.fit_transform(X[ord_cols].astype(object))
        return X

    def _apply_ordinal_encoder(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted OrdinalEncoder to ordinal columns."""
        if not self._fitted_ordinal_cols:
            return X
        present = _present(self._fitted_ordinal_cols, X)
        X[present] = self._ordinal_encoder.transform(X[present].astype(object))
        return X

    def _recompute_sdq_derived(self, X: pd.DataFrame) -> pd.DataFrame:
        """Recompute derived SDQ/WEMWBS features from imputed source columns.

        Called after MICE and mode imputation so that all source columns are
        already filled.  Overwrites any NaN values left from pre-imputation
        feature engineering.
        """
        subs = self._cfg._schema.get("sdq_subscales", {})  # noqa: SLF001
        parent_int = subs.get("parent_internalising", [])
        self_int   = subs.get("self_internalising", [])
        parent_items = subs.get("parent_sdq_items", [])
        self_items   = subs.get("self_sdq_items", [])

        if "sdq_internalising" in X.columns:
            cols = [c for c in self_int if c in X.columns]
            if cols:
                X["sdq_internalising"] = X[cols].sum(axis=1)

        if "psdq_internalising" in X.columns:
            cols = [c for c in parent_int if c in X.columns]
            if cols:
                X["psdq_internalising"] = X[cols].sum(axis=1)

        if "parent_self_discrepancy" in X.columns:
            p_cols = [c for c in parent_items if c in X.columns]
            s_cols = [c for c in self_items   if c in X.columns]
            if p_cols and s_cols:
                X["parent_self_discrepancy"] = (
                    X[p_cols].sum(axis=1) - X[s_cols].sum(axis=1)
                )

        if "wellbeing_low_flag" in X.columns and "wellbeing_wemwbs" in X.columns:
            X["wellbeing_low_flag"] = (X["wellbeing_wemwbs"] < 40.0).astype(float)

        if "any_bullying" in X.columns:
            b  = X.get("bullied_flag",      pd.Series(0, index=X.index))
            cb = X.get("cyberbullied_flag",  pd.Series(0, index=X.index))
            X["any_bullying"] = ((b == 1) | (cb == 1)).astype(float)

        return X

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure X has exactly the columns seen at fit time, in the same order."""
        expected = self._feature_names_out
        missing = [c for c in expected if c not in X.columns]
        if missing:
            logger.warning("Aligning columns: adding %d missing columns as NaN: %s", len(missing), missing)
            for c in missing:
                X[c] = np.nan
        extra = [c for c in X.columns if c not in expected]
        if extra:
            logger.debug("Aligning columns: dropping %d extra columns: %s", len(extra), extra)
        return X[expected]


# ── SMOTE helper ─────────────────────────────────────────────────────────

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to the training set only.

    SMOTE must never be applied to validation or test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Preprocessed training features.
    y_train : pd.Series
        Training labels.
    cfg : Config
        Configuration object (provides SMOTE parameters).

    Returns
    -------
    X_resampled, y_resampled : pd.DataFrame, pd.Series
    """
    from imblearn.over_sampling import SMOTE

    n_minority = y_train.sum()
    n_majority = len(y_train) - n_minority

    if n_minority < cfg.smote_k_neighbors + 1:
        logger.warning(
            "Too few minority samples (%d) for SMOTE (k=%d). Skipping SMOTE.",
            n_minority, cfg.smote_k_neighbors,
        )
        return X_train, y_train

    smote = SMOTE(
        sampling_strategy=cfg.smote_sampling_strategy,
        k_neighbors=cfg.smote_k_neighbors,
        random_state=cfg.random_seed,
    )
    feature_names = X_train.columns.tolist()
    X_arr, y_arr = smote.fit_resample(X_train.values, y_train.values)

    X_resampled = pd.DataFrame(X_arr, columns=feature_names)
    y_resampled = pd.Series(y_arr, name=y_train.name)

    logger.info(
        "SMOTE applied: %d → %d training rows (at-risk: %d → %d)",
        len(y_train), len(y_resampled),
        int(n_minority), int(y_resampled.sum()),
    )
    return X_resampled, y_resampled
