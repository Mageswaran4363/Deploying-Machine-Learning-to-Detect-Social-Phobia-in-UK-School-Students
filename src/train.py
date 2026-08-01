"""
Training pipeline — true nested cross-validation with Optuna hyperparameter search.

Nested CV design (ml-development-plan.md Section 4.3):
  Outer loop — 5-fold StratifiedGroupKFold by school_id.
    Each fold holds out one school as the outer test set.  Metrics aggregated
    across all outer folds (mean ± SD) are the figures cited in any publication.
  Inner loop — 3-fold StratifiedGroupKFold within the outer training set.
    Optuna TPE sampler tunes hyperparameters by maximising Recall (primary
    optimisation criterion; false negatives carry greater harm in a safeguarding
    context).

Pipeline stages:
  1. Load active dataset → validate schema → apply exclusion criteria.
  2. Feature engineering (stateless FeatureEngineer).
  3. Reserve held-out test school (school-isolated; touched once at step 10).
  4. Form CV pool (remaining data with real school groups).
  5. Outer loop — for each outer fold:
       a. Fit Preprocessor on outer training split only.
       b. Run inner Optuna tuning per model on pre-SMOTE outer training data.
       c. Train each model on SMOTE-augmented outer training data.
       d. Evaluate each model on the outer validation split.
       e. Record FoldReport.
  6. Aggregate outer-fold metrics: mean ± SD per model.
  7. Select best model type by mean composite SelectionScore.
  8. Retrain final model on full CV pool using best params from highest-scoring fold.
  9. Derive classification threshold: mean of per-fold thresholds for best model.
 10. Check hard deployment gates against mean outer-CV metrics.
 11. Evaluate once on held-out test school.
 12. Save artefacts and update config/model_registry.yaml.

Usage
-----
>>> from src.config import Config
>>> from src.train import run_training_pipeline
>>> cfg = Config()
>>> results = run_training_pipeline(cfg)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import Config
from src.data_loader import apply_exclusion_criteria, load_active_dataset
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor, apply_smote
from src.schema_validator import SchemaValidator

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Result dataclasses ────────────────────────────────────────────────────

@dataclass
class FoldReport:
    """Structured result for one outer CV fold.

    Attributes
    ----------
    fold_idx : int
        0-based outer fold index.
    test_school_ids : list[str]
        School(s) held out as the outer test set for this fold.
    n_test : int
        Number of records in the outer test split.
    n_positive_test : int
        At-risk students in the outer test split.
    model_metrics : dict
        ``{model_name: {metric: value}}`` evaluated on the outer test split.
    model_params : dict
        ``{model_name: best_hyperparams_dict}`` from inner Optuna tuning.
    model_thresholds : dict
        ``{model_name: optimised_threshold}`` from outer validation split.
    model_composites : dict
        ``{model_name: composite_score}`` on the outer test split.
    best_model_name : str
        Model with the highest composite score in this fold.
    best_composite : float
        Composite score of the best model in this fold.
    """

    fold_idx: int
    test_school_ids: list[str]
    n_test: int
    n_positive_test: int
    model_metrics: dict[str, dict] = field(default_factory=dict)
    model_params: dict[str, dict] = field(default_factory=dict)
    model_thresholds: dict[str, float] = field(default_factory=dict)
    model_composites: dict[str, float] = field(default_factory=dict)
    best_model_name: str = ""
    best_composite: float = 0.0

    def summary(self) -> str:
        """Return a formatted table of all model metrics for this fold."""
        header = (
            f"\n{'─' * 72}\n"
            f"  Outer Fold {self.fold_idx + 1} | "
            f"Test school(s): {self.test_school_ids} | "
            f"n={self.n_test} ({self.n_positive_test} at-risk)\n"
            f"{'─' * 72}\n"
            f"  {'Model':<22} {'Recall':>7} {'PR-AUC':>7} {'F1':>7} "
            f"{'AUC-ROC':>8} {'Composite':>10}\n"
            f"  {'─'*22} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*10}"
        )
        rows = []
        for name, m in self.model_metrics.items():
            marker = " ◀" if name == self.best_model_name else ""
            rows.append(
                f"  {name:<22} "
                f"{m.get('recall', float('nan')):>7.3f} "
                f"{m.get('pr_auc', float('nan')):>7.3f} "
                f"{m.get('f1', float('nan')):>7.3f} "
                f"{m.get('auc_roc', float('nan')):>8.3f} "
                f"{self.model_composites.get(name, float('nan')):>10.4f}"
                f"{marker}"
            )
        return header + "\n" + "\n".join(rows) + f"\n{'─' * 72}"


@dataclass
class NestedCVResults:
    """Aggregate results across all outer CV folds.

    Attributes
    ----------
    n_outer_folds : int
        Number of outer folds actually run.
    fold_reports : list[FoldReport]
        One FoldReport per outer fold.
    model_summary : dict
        ``{model_name: {"mean_metrics": {...}, "std_metrics": {...},
        "mean_composite": float, "std_composite": float}}``
    best_model_name : str
        Model selected by mean composite score (tie-broken by interpretability).
    best_fold_idx : int
        Outer fold index where the best model achieved its highest composite.
    """

    n_outer_folds: int
    fold_reports: list[FoldReport] = field(default_factory=list)
    model_summary: dict = field(default_factory=dict)
    best_model_name: str = ""
    best_fold_idx: int = 0

    def summary(self) -> str:
        """Return a formatted aggregate table (mean ± SD across folds)."""
        header = (
            f"\n{'═' * 80}\n"
            f"  NESTED CV RESULTS  |  {self.n_outer_folds} outer folds\n"
            f"{'═' * 80}\n"
            f"  {'Model':<22} "
            f"{'Recall':>13} {'PR-AUC':>13} {'F1':>13} "
            f"{'AUC-ROC':>13} {'Composite':>13}\n"
            f"  {'─'*22} {'─'*13} {'─'*13} {'─'*13} {'─'*13} {'─'*13}"
        )
        rows = []
        for name, s in self.model_summary.items():
            mm = s["mean_metrics"]
            sm = s["std_metrics"]
            mc = s["mean_composite"]
            sc = s["std_composite"]
            marker = "  ◀ SELECTED" if name == self.best_model_name else ""

            def fmt(mean_val: float, std_val: float) -> str:
                return f"{mean_val:.3f}±{std_val:.3f}"

            rows.append(
                f"  {name:<22} "
                f"{fmt(mm.get('recall', 0), sm.get('recall', 0)):>13} "
                f"{fmt(mm.get('pr_auc', 0), sm.get('pr_auc', 0)):>13} "
                f"{fmt(mm.get('f1', 0), sm.get('f1', 0)):>13} "
                f"{fmt(mm.get('auc_roc', 0), sm.get('auc_roc', 0)):>13} "
                f"{fmt(mc, sc):>13}"
                f"{marker}"
            )
        return header + "\n" + "\n".join(rows) + f"\n{'═' * 80}"


# ── Candidate model catalogue ─────────────────────────────────────────────

def _build_model_catalogue(cfg: Config, scale_pos_weight: float = 9.0) -> dict:
    """Return all candidate models with their Optuna search space definitions.

    MLP is excluded: with n < 400 the network is underpowered (plan Section 6.1).
    All models include class_weight or scale_pos_weight to address imbalance.

    Parameters
    ----------
    cfg : Config
    scale_pos_weight : float
        XGBoost's imbalance weight (n_negative / n_positive in training fold).
    """
    seed = cfg.random_seed
    return {
        "LogisticRegression": {
            "model_cls": LogisticRegression,
            "fixed_params": {
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": seed,
                "solver": "lbfgs",
                "l1_ratio": 0,  # equivalent to deprecated penalty='l2'
            },
            "search_space": lambda trial: {
                "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            },
        },
        "DecisionTree": {
            "model_cls": DecisionTreeClassifier,
            "fixed_params": {"class_weight": "balanced", "random_state": seed},
            "search_space": lambda trial: {
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            },
        },
        "RandomForest": {
            "model_cls": RandomForestClassifier,
            "fixed_params": {
                "class_weight": "balanced_subsample",
                "random_state": seed,
                "n_jobs": -1,
            },
            "search_space": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.5]
                ),
            },
        },
        "XGBoost": {
            "model_cls": XGBClassifier,
            "fixed_params": {
                "scale_pos_weight": scale_pos_weight,
                "random_state": seed,
                "eval_metric": "logloss",
                "verbosity": 0,
            },
            "search_space": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            },
        },
        "SVM": {
            # CalibratedClassifierCV replaces SVC(probability=True) deprecated in sklearn 1.9
            "model_cls": lambda **kw: CalibratedClassifierCV(SVC(**kw), ensemble=False),
            "fixed_params": {
                "class_weight": "balanced",
                "random_state": seed,
            },
            "search_space": lambda trial: {
                "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            },
        },
    }


# ── Composite selection score ─────────────────────────────────────────────

def _selection_score(metrics: dict, weights: dict) -> float:
    """Compute the weighted composite model selection score.

    Formula: 0.45×Recall + 0.25×PR-AUC + 0.20×F1 + 0.10×AUC-ROC
    (weights from model_registry.yaml / ml-development-plan.md Section 6.3).
    """
    return sum(weights.get(k, 0.0) * v for k, v in metrics.items()
               if not np.isnan(v))


# ── Threshold optimisation ────────────────────────────────────────────────

def optimise_threshold(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    recall_minimum: float,
) -> float:
    """Find the threshold that maximises F1 subject to Recall ≥ recall_minimum.

    Parameters
    ----------
    model : fitted classifier with ``predict_proba``.
    X_val : pd.DataFrame
    y_val : pd.Series
    recall_minimum : float
        Hard lower bound on Recall (default 0.85 per deployment gate).

    Returns
    -------
    float
        Optimal threshold in [0, 1].
    """
    y_prob = model.predict_proba(X_val.values)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val.values.astype(int), y_prob)

    # precision_recall_curve returns n+1 points; thresholds has length n
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
    valid_mask = rec[:-1] >= recall_minimum

    if not valid_mask.any():
        logger.warning(
            "No threshold achieves Recall >= %.2f; returning lowest available.",
            recall_minimum,
        )
        return float(thresholds[0])

    best_idx = int(np.argmax(f1[valid_mask]))
    optimal = float(thresholds[valid_mask][best_idx])
    logger.info(
        "Threshold optimised: %.4f  (Recall=%.3f  Precision=%.3f  F1=%.3f)",
        optimal,
        float(rec[:-1][valid_mask][best_idx]),
        float(prec[:-1][valid_mask][best_idx]),
        float(f1[valid_mask][best_idx]),
    )
    return optimal


# ── Per-instance metrics helper ───────────────────────────────────────────

def _compute_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Evaluate a fitted model on X/y using the given threshold.

    Returns a dict with the four SelectionScore metrics.  AUC-ROC is NaN when
    only one class is present (logged as a warning, not an error).
    """
    y_np = y.values.astype(int)
    y_prob = model.predict_proba(X.values)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc = (
        float(roc_auc_score(y_np, y_prob))
        if len(np.unique(y_np)) > 1
        else float("nan")
    )
    if np.isnan(auc):
        logger.warning("AUC-ROC undefined for fold with single class present.")

    return {
        "recall": float(recall_score(y_np, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_np, y_prob))
        if len(np.unique(y_np)) > 1
        else float("nan"),
        "f1": float(f1_score(y_np, y_pred, zero_division=0)),
        "auc_roc": auc,
    }


# ── Inner CV / Optuna objective ───────────────────────────────────────────

def _tune_model(
    model_name: str,
    catalogue_entry: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_trials: int,
    n_inner_folds: int,
    seed: int,
) -> dict:
    """Run Optuna Bayesian optimisation in the inner CV loop.

    Receives **pre-SMOTE** outer training data.  Class weights on all models
    handle imbalance at the inner-fold level; SMOTE is applied only when
    training the final model for each outer fold.  This avoids passing a
    SMOTE-expanded array whose row count no longer matches ``groups_train``.

    Parameters
    ----------
    X_train : np.ndarray
        Pre-SMOTE preprocessed outer training features.
    y_train : np.ndarray
        Pre-SMOTE outer training labels.
    groups_train : np.ndarray
        School IDs for the outer training samples (same length as X_train).
    seed : int
        Per-fold seed for Optuna TPE sampler reproducibility.

    Returns
    -------
    dict
        Best hyperparameters found by Optuna.
    """
    fixed = catalogue_entry["fixed_params"]
    search_space = catalogue_entry["search_space"]
    model_cls = catalogue_entry["model_cls"]

    n_inner = min(n_inner_folds, len(np.unique(groups_train)))
    if n_inner < 2:
        logger.warning(
            "%s: only %d school(s) in outer train fold — cannot run inner CV; "
            "using default hyperparameters.",
            model_name, n_inner,
        )
        return {}
    if n_inner < n_inner_folds:
        logger.warning(
            "%s: only %d school(s) in outer train fold; "
            "reducing inner folds from %d to %d.",
            model_name, n_inner, n_inner_folds, n_inner,
        )

    inner_cv = StratifiedGroupKFold(
        n_splits=n_inner, shuffle=True, random_state=seed
    )

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed, **search_space(trial)}
        model = model_cls(**params)
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=inner_cv,
            groups=groups_train,
            scoring="recall",
            n_jobs=-1,
            error_score=0.0,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(
        "%s inner tuning: best Recall=%.4f | params=%s",
        model_name,
        study.best_value,
        study.best_params,
    )
    return study.best_params


# ── Test school separation ────────────────────────────────────────────────

def _separate_test_school(
    df: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.Series, pd.Series,
           pd.DataFrame, pd.Series]:
    """Reserve one school as the held-out test set; return CV pool and test.

    Uses the first fold of ``StratifiedGroupKFold`` to isolate one school.
    The test set is never touched until the final evaluation step.

    Returns
    -------
    X_pool, y_pool, groups_pool : DataFrame / Series
        Feature matrix, labels, and school IDs for the CV pool.
    X_test, y_test : DataFrame / Series
        Held-out test features and labels.
    """
    target = cfg.primary_target_col
    group_col = cfg.group_col
    feature_cols = [c for c in cfg.model_feature_cols if c in df.columns]

    y = df[target].astype(int)
    groups = df[group_col]
    X = df[feature_cols]

    n_schools = groups.nunique()
    n_splits = min(cfg.cv_outer_folds, n_schools)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=cfg.random_seed
    )
    pool_idx, test_idx = next(iter(sgkf.split(X, y, groups=groups)))

    X_pool    = X.iloc[pool_idx].reset_index(drop=True)
    y_pool    = y.iloc[pool_idx].reset_index(drop=True)
    groups_pool = groups.iloc[pool_idx].reset_index(drop=True)

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    test_schools = groups.iloc[test_idx].unique().tolist()
    logger.info(
        "Test school(s) reserved: %s  |  pool n=%d  test n=%d",
        test_schools, len(X_pool), len(X_test),
    )
    return X_pool, y_pool, groups_pool, X_test, y_test


# ── True nested cross-validation ──────────────────────────────────────────

def _run_nested_cv(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    groups_pool: pd.Series,
    cfg: Config,
) -> NestedCVResults:
    """Run the full outer CV loop, collecting one FoldReport per fold.

    For each outer fold:
      1. Fit a fresh Preprocessor on the outer training split.
      2. Run inner Optuna tuning per model on pre-SMOTE outer training data.
      3. Train each model on SMOTE-augmented outer training data.
      4. Optimise threshold and compute metrics on the outer validation split.
      5. Record results in a FoldReport.

    Parameters
    ----------
    X_pool : pd.DataFrame
        Feature matrix for the entire CV pool (no test school).
    y_pool : pd.Series
        Labels for the CV pool.
    groups_pool : pd.Series
        School IDs for the CV pool.
    cfg : Config

    Returns
    -------
    NestedCVResults
        Fold-level reports and aggregated mean ± SD summary.
    """
    n_schools = groups_pool.nunique()
    n_outer = min(cfg.cv_outer_folds, n_schools)
    if n_outer < cfg.cv_outer_folds:
        logger.warning(
            "Only %d school(s) in pool; reducing outer folds from %d to %d.",
            n_schools, cfg.cv_outer_folds, n_outer,
        )

    outer_cv = StratifiedGroupKFold(
        n_splits=n_outer, shuffle=True, random_state=cfg.random_seed
    )

    fold_reports: list[FoldReport] = []

    for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(
        outer_cv.split(X_pool, y_pool, groups=groups_pool)
    ):
        logger.info("── Outer fold %d / %d ──", fold_idx + 1, n_outer)

        # ── Slice outer train and val ─────────────────────────────────────
        X_outer_train = X_pool.iloc[outer_train_idx].reset_index(drop=True)
        X_outer_val   = X_pool.iloc[outer_val_idx].reset_index(drop=True)
        y_outer_train = y_pool.iloc[outer_train_idx].reset_index(drop=True)
        y_outer_val   = y_pool.iloc[outer_val_idx].reset_index(drop=True)
        groups_outer  = groups_pool.iloc[outer_train_idx].reset_index(drop=True)
        test_schools  = groups_pool.iloc[outer_val_idx].unique().tolist()

        logger.info(
            "  Fold %d: outer_train n=%d (%.1f%% +)  outer_val n=%d (%.1f%% +)  "
            "val_schools=%s",
            fold_idx + 1,
            len(y_outer_train), y_outer_train.mean() * 100,
            len(y_outer_val),   y_outer_val.mean() * 100,
            test_schools,
        )

        # ── Preprocess (fit on outer_train only) ─────────────────────────
        fold_preprocessor = Preprocessor(cfg)
        X_outer_train_proc = fold_preprocessor.fit_transform(
            X_outer_train, groups=groups_outer
        )
        X_outer_val_proc = fold_preprocessor.transform(X_outer_val)

        # ── SMOTE on outer training fold ──────────────────────────────────
        X_outer_smote, y_outer_smote = apply_smote(
            X_outer_train_proc, y_outer_train, cfg
        )

        # ── XGBoost scale_pos_weight from pre-SMOTE distribution ─────────
        n_pos = int(y_outer_train.sum())
        n_neg = len(y_outer_train) - n_pos
        spw   = max(1.0, n_neg / max(n_pos, 1))
        catalogue = _build_model_catalogue(cfg, scale_pos_weight=spw)
        weights   = cfg.selection_weights

        fold_model_metrics: dict[str, dict]  = {}
        fold_model_params:  dict[str, dict]  = {}
        fold_model_thresholds: dict[str, float] = {}
        fold_model_composites: dict[str, float] = {}

        for model_name, entry in catalogue.items():
            # Inner Optuna tuning — pre-SMOTE data; fold-specific seed
            best_params = _tune_model(
                model_name=model_name,
                catalogue_entry=entry,
                X_train=X_outer_train_proc.values,   # pre-SMOTE ← correct
                y_train=y_outer_train.values,         # pre-SMOTE ← correct
                groups_train=groups_outer.values,
                n_trials=cfg.optuna_n_trials,
                n_inner_folds=cfg.cv_inner_folds,
                seed=cfg.random_seed + fold_idx,      # fold-specific seed
            )

            # Train on SMOTE-augmented outer training data
            params = {**entry["fixed_params"], **best_params}
            model  = entry["model_cls"](**params)
            model.fit(X_outer_smote.values, y_outer_smote.values)

            # Threshold optimisation on outer_val
            threshold = optimise_threshold(
                model, X_outer_val_proc, y_outer_val, cfg.recall_minimum
            )

            # Outer-fold evaluation
            metrics   = _compute_metrics(model, X_outer_val_proc, y_outer_val, threshold)
            composite = _selection_score(metrics, weights)

            fold_model_metrics[model_name]    = metrics
            fold_model_params[model_name]     = best_params
            fold_model_thresholds[model_name] = threshold
            fold_model_composites[model_name] = composite

            logger.info(
                "  Fold %d | %s | composite=%.4f "
                "recall=%.3f pr_auc=%.3f f1=%.3f auc_roc=%.3f",
                fold_idx + 1, model_name, composite,
                metrics["recall"], metrics.get("pr_auc", float("nan")),
                metrics["f1"],     metrics.get("auc_roc", float("nan")),
            )

        # Best model for this fold (composite score, tie-break: interpretability)
        best_name = max(
            fold_model_composites,
            key=lambda k: (
                fold_model_composites[k],
                -_interpretability_rank(k, cfg),
            ),
        )

        report = FoldReport(
            fold_idx=fold_idx,
            test_school_ids=test_schools,
            n_test=len(y_outer_val),
            n_positive_test=int(y_outer_val.sum()),
            model_metrics=fold_model_metrics,
            model_params=fold_model_params,
            model_thresholds=fold_model_thresholds,
            model_composites=fold_model_composites,
            best_model_name=best_name,
            best_composite=fold_model_composites[best_name],
        )
        fold_reports.append(report)
        logger.info("%s", report.summary())

    # ── Aggregate across folds ────────────────────────────────────────────
    model_summary = _aggregate_fold_results(fold_reports)
    best_model_name, best_fold_idx = _select_best_model(model_summary, fold_reports, cfg)

    results = NestedCVResults(
        n_outer_folds=n_outer,
        fold_reports=fold_reports,
        model_summary=model_summary,
        best_model_name=best_model_name,
        best_fold_idx=best_fold_idx,
    )
    logger.info("%s", results.summary())
    return results


# ── Aggregate helper ──────────────────────────────────────────────────────

def _aggregate_fold_results(
    fold_reports: list[FoldReport],
) -> dict[str, dict]:
    """Compute mean ± SD across outer folds for every model and metric.

    Returns
    -------
    dict
        ``{model_name: {
              "mean_metrics": {...},
              "std_metrics": {...},
              "mean_composite": float,
              "std_composite": float,
              "fold_composites": [float, ...],
          }}``
    """
    model_names = list(fold_reports[0].model_metrics.keys())
    summary: dict[str, dict] = {}

    for model_name in model_names:
        fold_metrics_list = [fr.model_metrics[model_name] for fr in fold_reports]
        fold_composites   = [fr.model_composites[model_name] for fr in fold_reports]
        metric_keys = list(fold_metrics_list[0].keys())

        # Use nanmean / nanstd so a single fold's NaN AUC doesn't kill the mean
        mean_metrics = {
            k: float(np.nanmean([m[k] for m in fold_metrics_list]))
            for k in metric_keys
        }
        std_metrics = {
            k: float(np.nanstd([m[k] for m in fold_metrics_list], ddof=1))
            if len(fold_metrics_list) > 1 else 0.0
            for k in metric_keys
        }

        summary[model_name] = {
            "mean_metrics":   mean_metrics,
            "std_metrics":    std_metrics,
            "mean_composite": float(np.mean(fold_composites)),
            "std_composite":  float(np.std(fold_composites, ddof=1))
            if len(fold_composites) > 1 else 0.0,
            "fold_composites": fold_composites,
        }

    return summary


# ── Model selection ───────────────────────────────────────────────────────

def _select_best_model(
    model_summary: dict,
    fold_reports: list[FoldReport],
    cfg: Config,
) -> tuple[str, int]:
    """Select the best model type by mean composite score.

    Tie-break: prefer the more interpretable model when scores are within 0.02.

    Returns
    -------
    best_model_name : str
    best_fold_idx : int
        Outer fold index where the best model scored highest (used to retrieve
        the best hyperparameters for final retraining).
    """
    best_name = max(
        model_summary,
        key=lambda k: (
            model_summary[k]["mean_composite"],
            -_interpretability_rank(k, cfg),
        ),
    )

    # Find the fold where the best model had its highest composite
    best_fold_idx = int(
        np.argmax([
            fr.model_composites.get(best_name, -1.0)
            for fr in fold_reports
        ])
    )

    logger.info(
        "Selected model: %s | mean composite=%.4f ± %.4f | "
        "best params from fold %d",
        best_name,
        model_summary[best_name]["mean_composite"],
        model_summary[best_name]["std_composite"],
        best_fold_idx + 1,
    )
    return best_name, best_fold_idx


# ── Main training pipeline ────────────────────────────────────────────────

def run_training_pipeline(cfg: Config) -> dict:
    """Execute the full ML training pipeline and return evaluation results.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object.

    Returns
    -------
    dict with keys:
        ``best_model_name``, ``best_model``, ``preprocessor``, ``threshold``,
        ``cv_results`` (NestedCVResults), ``test_metrics``, ``artefact_paths``,
        ``feature_names``.
    """
    t_start = time.time()
    logger.info("=== Training pipeline START (dataset: %s) ===", cfg.active_dataset_key)

    # ── 1. Load and validate ──────────────────────────────────────────────
    df, _ = load_active_dataset(cfg)

    validator = SchemaValidator(cfg)
    report = validator.validate(df, cfg.active_dataset_key)
    if not report.passed:
        raise ValueError(
            f"Schema validation failed for '{cfg.active_dataset_key}'.\n"
            + report.summary()
        )

    df = apply_exclusion_criteria(df, cfg)

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} records remain after exclusions; "
            "need ≥ 50 for reliable cross-validation."
        )

    # ── 2. Feature engineering ────────────────────────────────────────────
    fe = FeatureEngineer(cfg)
    df = fe.transform(df)

    # ── 3. Reserve held-out test school ──────────────────────────────────
    X_pool, y_pool, groups_pool, X_test, y_test = _separate_test_school(df, cfg)

    # ── 4–6. True nested CV over pool ─────────────────────────────────────
    cv_results = _run_nested_cv(X_pool, y_pool, groups_pool, cfg)

    # ── 7. Check hard deployment gates (against mean CV metrics) ──────────
    best_mean_metrics = cv_results.model_summary[cv_results.best_model_name][
        "mean_metrics"
    ]
    _check_deployment_gates(best_mean_metrics, cfg)

    # ── 8. Retrain final model on full CV pool ────────────────────────────
    # Preprocessor fit on the entire pool; best hyperparams from top-scoring fold
    final_preprocessor = Preprocessor(cfg)
    X_pool_proc = final_preprocessor.fit_transform(X_pool, groups=groups_pool)
    X_test_proc = final_preprocessor.transform(X_test)

    X_pool_smote, y_pool_smote = apply_smote(X_pool_proc, y_pool, cfg)

    best_fold = cv_results.fold_reports[cv_results.best_fold_idx]
    best_entry = _build_model_catalogue(cfg)[cv_results.best_model_name]
    best_params = best_fold.model_params[cv_results.best_model_name]
    final_params = {**best_entry["fixed_params"], **best_params}

    final_model = best_entry["model_cls"](**final_params)
    final_model.fit(X_pool_smote.values, y_pool_smote.values)
    logger.info(
        "Final model retrained on full pool: %s (%d rows after SMOTE).",
        cv_results.best_model_name, len(y_pool_smote),
    )

    # ── 9. Classification threshold ───────────────────────────────────────
    # Use the mean of per-fold thresholds for the selected model.
    threshold = float(np.mean([
        fr.model_thresholds[cv_results.best_model_name]
        for fr in cv_results.fold_reports
    ]))
    logger.info("Classification threshold (mean of outer folds): %.4f", threshold)

    # ── 10. Final test evaluation (once only) ─────────────────────────────
    test_metrics = _compute_metrics(final_model, X_test_proc, y_test, threshold)
    test_metrics["precision"] = float(
        precision_score(
            y_test.values.astype(int),
            (final_model.predict_proba(X_test_proc.values)[:, 1] >= threshold).astype(int),
            zero_division=0,
        )
    )
    logger.info("Held-out test metrics: %s", test_metrics)

    # ── 11. Save artefacts ────────────────────────────────────────────────
    artefact_paths = _save_artefacts(
        model=final_model,
        preprocessor=final_preprocessor,
        model_name=cv_results.best_model_name,
        dataset_key=cfg.active_dataset_key,
        threshold=threshold,
        cv_results=cv_results,
        test_metrics=test_metrics,
        cfg=cfg,
    )

    t_elapsed = time.time() - t_start
    logger.info("=== Training pipeline COMPLETE in %.1f s ===", t_elapsed)

    return {
        "best_model_name": cv_results.best_model_name,
        "best_model": final_model,
        "preprocessor": final_preprocessor,
        "threshold": threshold,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "artefact_paths": artefact_paths,
        "feature_names": final_preprocessor.feature_names_out,
    }


# ── Utilities ─────────────────────────────────────────────────────────────

def _interpretability_rank(model_name: str, cfg: Config) -> int:
    """Lower rank = more interpretable = preferred in tie-break."""
    rank_list = cfg.interpretability_rank
    try:
        return rank_list.index(model_name)
    except ValueError:
        return len(rank_list)


def _check_deployment_gates(metrics: dict, cfg: Config) -> None:
    """Raise ValueError if any hard deployment gate is failed.

    Gates (from model_registry.yaml / ml-development-plan.md Section 6.3):
      Recall ≥ 0.85  |  AUC-ROC ≥ 0.80  |  PR-AUC ≥ 0.50
    """
    gates = {
        "recall":  cfg.recall_minimum,
        "auc_roc": cfg.auc_roc_minimum,
        "pr_auc":  cfg.pr_auc_minimum,
    }
    failures = [
        f"{metric} = {metrics.get(metric, 0.0):.4f} < {minimum}"
        for metric, minimum in gates.items()
        if metrics.get(metric, 0.0) < minimum
    ]
    if failures:
        raise ValueError(
            "Model failed hard deployment gate(s):\n"
            + "\n".join(f"  • {f}" for f in failures)
            + "\nReview dataset or adjust threshold before deploying."
        )


def _save_artefacts(
    model,
    preprocessor: Preprocessor,
    model_name: str,
    dataset_key: str,
    threshold: float,
    cv_results: NestedCVResults,
    test_metrics: dict,
    cfg: Config,
) -> dict[str, Path]:
    """Serialise trained artefacts and update config/model_registry.yaml.

    Stores mean ± SD CV performance (``cv_performance``) alongside held-out
    test performance (``test_performance``) so the registry captures the
    publication-quality metrics rather than a single-split estimate.
    """
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    registry_path = cfg._config_dir / "model_registry.yaml"  # noqa: SLF001
    with registry_path.open(encoding="utf-8") as fh:
        registry = yaml.safe_load(fh)

    existing = registry.get("models", {})
    version  = f"v{len(existing) + 1}"

    paths = {
        "model":        models_dir / f"model_{version}.pkl",
        "preprocessor": models_dir / f"preprocessor_{version}.pkl",
        "spin_imputer": models_dir / f"spin_imputer_{version}.pkl",
    }

    joblib.dump(model, paths["model"])
    preprocessor.save(paths["preprocessor"])
    joblib.dump(preprocessor._spin_imputer, paths["spin_imputer"])  # noqa: SLF001

    ms = cv_results.model_summary[model_name]
    registry.setdefault("models", {})[version] = {
        "model_name":               model_name,
        "model_path":               str(paths["model"]),
        "preprocessor_path":        str(paths["preprocessor"]),
        "spin_imputer_path":        str(paths["spin_imputer"]),
        "trained_on_dataset":       dataset_key,
        "trained_date":             pd.Timestamp.now().strftime("%Y-%m-%d"),
        "n_outer_cv_folds":         cv_results.n_outer_folds,
        "classification_threshold": round(threshold, 4),
        "cv_performance": {
            "mean": {k: round(v, 4) for k, v in ms["mean_metrics"].items()},
            "std":  {k: round(v, 4) for k, v in ms["std_metrics"].items()},
            "mean_composite": round(ms["mean_composite"], 4),
            "std_composite":  round(ms["std_composite"],  4),
        },
        "test_performance": {k: round(v, 4) for k, v in test_metrics.items()},
        "status": "active",
    }
    registry["active_model"] = version

    with registry_path.open("w", encoding="utf-8") as fh:
        yaml.dump(registry, fh, default_flow_style=False, allow_unicode=True)

    logger.info(
        "Artefacts saved: model=%s  version=%s  registry=%s",
        model_name, version, registry_path,
    )
    return paths
