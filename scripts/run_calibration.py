"""
Platt Scaling calibration for model v4.

Problem: model v4 ECE = 0.43 (target < 0.05). The predicted probabilities are
distorted by class_weight='balanced' — they do not reflect the true 2% base rate.

Approach:
  1. Reproduce the original test / CV-pool split (same seed → same test school).
  2. Split the CV pool 80/20 (stratified by school + label).
  3. Retrain a LogisticRegression (same hyperparameters as v4) on the 80%.
  4. Fit CalibratedClassifierCV(cv='prefit', method='sigmoid') on the 20%.
  5. Re-derive the classification threshold on the calibration set.
  6. Evaluate: ECE / Brier / recall / precision before vs. after on the test set.
  7. Save the calibrated model as model_v5.pkl and update model_registry.yaml.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration import PlattCalibratedModel
from src.config import Config
from src.feature_engineering import FeatureEngineer
from src.preprocessing import apply_smote

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REGISTRY_PATH = Path("config/model_registry.yaml")


# ── Helpers ───────────────────────────────────────────────────────────────

def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    weights = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    weights = weights[weights > 0] / len(y_prob)
    diffs   = np.abs(prob_true - prob_pred)
    return float(np.sum(weights * diffs[:len(weights)]))


def _optimise_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_minimum: float = 0.85,
) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
    valid = rec[:-1] >= recall_minimum
    if not valid.any():
        return float(thresholds[0])
    return float(thresholds[valid][np.argmax(f1[valid])])


def _report(label: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred  = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    brier     = brier_score_loss(y_true, y_prob)
    ece_val   = _ece(y_true, y_prob)
    pr_auc    = average_precision_score(y_true, y_prob)
    auc_roc   = roc_auc_score(y_true, y_prob)
    print(f"\n── {label} ──")
    print(f"  Threshold : {threshold:.4f}")
    print(f"  Recall    : {recall:.4f}   (TP={tp}  FN={fn})")
    print(f"  Precision : {precision:.4f}  (FP={fp}  TN={tn})")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"  Brier     : {brier:.4f}   (target < 0.10)")
    print(f"  ECE       : {ece_val:.4f}   (target < 0.05)  {'✅' if ece_val < 0.05 else '❌'}")
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "pr_auc": round(pr_auc, 4),
        "auc_roc": round(auc_roc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece_val, 4),
        "threshold": round(threshold, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    cfg  = Config()
    prep = joblib.load("models/preprocessor_v4.pkl")

    # ── Load and prepare full dataset ─────────────────────────────────────
    logger.info("Loading real_v1.csv …")
    df_raw = pd.read_csv("data/raw/real_v1.csv")
    fe     = FeatureEngineer(cfg)
    df_fe  = fe.transform(df_raw.copy())

    feature_cols = [c for c in cfg.model_feature_cols if c in df_fe.columns]
    target_col   = cfg.primary_target_col
    group_col    = cfg.group_col

    X      = df_fe[feature_cols]
    y      = df_fe[target_col].astype(int)
    groups = df_fe[group_col]

    # ── Reproduce original test / pool split (same seed as training) ──────
    logger.info("Carving held-out test school (same split as v4 training) …")
    n_splits = min(cfg.cv_outer_folds, groups.nunique())
    sgkf     = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=cfg.random_seed
    )
    pool_idx, test_idx = next(iter(sgkf.split(X, y, groups=groups)))

    X_pool = X.iloc[pool_idx].reset_index(drop=True)
    y_pool = y.iloc[pool_idx].reset_index(drop=True)
    g_pool = groups.iloc[pool_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    logger.info("Pool n=%d (%d+)  Test n=%d (%d+)",
                len(y_pool), y_pool.sum(), len(y_test), y_test.sum())

    # ── Split pool → 80% train  +  20% calibration ───────────────────────
    logger.info("Splitting CV pool 80/20 for calibration …")
    calib_cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=cfg.random_seed + 99
    )
    train_idx, calib_idx = next(iter(calib_cv.split(X_pool, y_pool, groups=g_pool)))

    X_train = X_pool.iloc[train_idx].reset_index(drop=True)
    y_train = y_pool.iloc[train_idx].reset_index(drop=True)
    X_calib = X_pool.iloc[calib_idx].reset_index(drop=True)
    y_calib = y_pool.iloc[calib_idx].reset_index(drop=True)

    logger.info("Train n=%d (%d+)  Calib n=%d (%d+)",
                len(y_train), y_train.sum(), len(y_calib), y_calib.sum())

    # ── Preprocess (use fitted preprocessor_v4 — no refitting) ───────────
    X_train_proc = prep.transform(X_train)
    X_calib_proc = prep.transform(X_calib)
    X_test_proc  = prep.transform(X_test)

    # ── Baseline: model v4 on test set ───────────────────────────────────
    logger.info("Evaluating baseline (model v4) …")
    model_v4 = joblib.load("models/model_v4.pkl")
    prob_v4  = model_v4.predict_proba(X_test_proc.values)[:, 1]
    baseline = _report("BEFORE calibration (model v4)", y_test.values, prob_v4, threshold=0.438)

    # ── Train base model (same hyperparams as v4) on 80% of pool ─────────
    logger.info("Training base LogisticRegression on 80%% of pool …")
    # Apply SMOTE to training split only
    X_smote, y_smote = apply_smote(X_train_proc, y_train, cfg)

    base_model = LogisticRegression(
        C=model_v4.C,
        l1_ratio=model_v4.l1_ratio,
        solver=model_v4.solver,
        max_iter=model_v4.max_iter,
        class_weight=model_v4.class_weight,
        random_state=cfg.random_seed,
    )
    base_model.fit(X_smote.values, y_smote.values)

    # ── Fit Platt Scaling on calibration set ─────────────────────────────
    logger.info("Fitting Platt Scaling (sigmoid) on calibration set …")
    calibrated_model = PlattCalibratedModel(base_model)
    calibrated_model.fit(X_calib_proc.values, y_calib.values)

    # ── Optimise threshold on calibration set ─────────────────────────────
    prob_calib = calibrated_model.predict_proba(X_calib_proc.values)[:, 1]
    threshold_v5 = _optimise_threshold(y_calib.values, prob_calib, cfg.recall_minimum)
    logger.info("New threshold (optimised on calibration set): %.4f", threshold_v5)

    # ── Evaluate calibrated model on test set ─────────────────────────────
    logger.info("Evaluating calibrated model (v5) on test set …")
    prob_v5 = calibrated_model.predict_proba(X_test_proc.values)[:, 1]
    calibrated_result = _report(
        "AFTER calibration (model v5)", y_test.values, prob_v5, threshold=threshold_v5
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n── Calibration improvement ──")
    ece_before = baseline["ece"]
    ece_after  = calibrated_result["ece"]
    print(f"  ECE:    {ece_before:.4f} → {ece_after:.4f}  "
          f"({'✅ improved' if ece_after < ece_before else '⚠ no improvement'})")
    print(f"  Brier:  {baseline['brier_score']:.4f} → {calibrated_result['brier_score']:.4f}")
    print(f"  Recall: {baseline['recall']:.4f} → {calibrated_result['recall']:.4f}")
    print(f"  PR-AUC: {baseline['pr_auc']:.4f} → {calibrated_result['pr_auc']:.4f}")

    if calibrated_result["recall"] < cfg.recall_minimum:
        logger.warning(
            "Calibrated model recall %.4f below gate %.2f — not saving as active.",
            calibrated_result["recall"], cfg.recall_minimum,
        )
        return

    # ── Save calibrated model as v5 ───────────────────────────────────────
    logger.info("Saving calibrated model as model_v5 …")
    joblib.dump(calibrated_model, "models/model_v5.pkl")

    # Preprocessor and spin imputer are unchanged — reuse v4 artefacts
    import shutil
    shutil.copy("models/preprocessor_v4.pkl", "models/preprocessor_v5.pkl")
    shutil.copy("models/spin_imputer_v4.pkl",  "models/spin_imputer_v5.pkl")

    # ── Update model_registry.yaml ────────────────────────────────────────
    logger.info("Updating model_registry.yaml …")
    with open(REGISTRY_PATH) as f:
        registry = yaml.safe_load(f)

    registry["active_model"] = "v5"
    registry["models"]["v5"] = {
        "model_name":    "LogisticRegression+PlattScaling",
        "model_path":    "models/model_v5.pkl",
        "preprocessor_path": "models/preprocessor_v5.pkl",
        "spin_imputer_path": "models/spin_imputer_v5.pkl",
        "trained_date":  "2026-07-31",
        "trained_on_dataset": "real_v1",
        "status":        "active",
        "n_outer_cv_folds": 5,
        "classification_threshold": round(threshold_v5, 4),
        "calibration_method": "platt_scaling_sigmoid",
        "calibration_note": (
            "Platt Scaling fitted on 20%% CV pool holdout. "
            "Base LR trained on 80%% CV pool with SMOTE. "
            "Same hyperparameters as v4 (C=0.001005, l1_ratio=0)."
        ),
        "test_performance": {
            "recall":    calibrated_result["recall"],
            "precision": calibrated_result["precision"],
            "pr_auc":    calibrated_result["pr_auc"],
            "auc_roc":   calibrated_result["auc_roc"],
            "brier_score": calibrated_result["brier_score"],
            "ece":       calibrated_result["ece"],
        },
    }

    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    logger.info("Done. Active model updated to v5.")
    logger.info("Artefacts: models/model_v5.pkl  models/preprocessor_v5.pkl")


if __name__ == "__main__":
    main()
