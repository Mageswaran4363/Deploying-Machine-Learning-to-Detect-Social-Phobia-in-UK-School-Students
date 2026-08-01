"""
Generate SHAP explainability outputs for model v4 on the full real_v1 dataset.

Produces:
  reports/figures/shap_global_importance.png  — bar chart of mean |SHAP|
  reports/figures/shap_beeswarm.png           — beeswarm summary plot
  reports/figures/shap_waterfall_case.png     — waterfall for one true positive
  reports/figures/surrogate_tree_rules.txt    — shallow surrogate tree rules
  reports/figures/shap_values.csv             — per-row SHAP values for UKDA deposit
  reports/figures/fairness_audit.csv          — recall parity by gender / age / SEN
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.evaluate import ModelEvaluator
from src.explainability import ExplainabilityEngine
from src.feature_engineering import FeatureEngineer
from src.preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = Config()

    # ── Load artefacts ────────────────────────────────────────────────────
    logger.info("Loading model v4 artefacts …")
    model     = joblib.load("models/model_v4.pkl")
    prep      = joblib.load("models/preprocessor_v4.pkl")

    # ── Prepare full dataset ──────────────────────────────────────────────
    logger.info("Loading and preprocessing real_v1.csv …")
    df_raw = pd.read_csv("data/raw/real_v1.csv")
    fe     = FeatureEngineer(cfg)
    df_fe  = fe.transform(df_raw.copy())

    feature_cols = [c for c in cfg.model_feature_cols if c in df_fe.columns]
    X_all = df_fe[feature_cols].copy()
    y_all = df_fe[cfg.primary_target_col].astype(int)

    X_proc = prep.transform(X_all)

    logger.info("Dataset: %d rows × %d features | positives: %d (%.1f%%)",
                len(X_proc), X_proc.shape[1], int(y_all.sum()), y_all.mean() * 100)

    # ── Build background sample for LinearExplainer ───────────────────────
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_proc), size=200, replace=False)
    background = X_proc.values[bg_idx]

    # ── Explainability engine ─────────────────────────────────────────────
    logger.info("Building ExplainabilityEngine …")
    engine = ExplainabilityEngine(
        cfg,
        model=model,
        feature_names=feature_cols,
        figures_dir=FIGURES_DIR,
        background_data=background,
    )
    logger.info("Explainer type: %s", type(engine._explainer).__name__)

    # ── Global SHAP importance ────────────────────────────────────────────
    logger.info("Computing global SHAP importance (n=%d) …", len(X_proc))
    importance_df = engine.global_shap(X_proc, save_plot=True)
    print("\n── Top 10 features by global SHAP importance ──")
    print(importance_df.head(10).to_string(index=False))
    importance_df.to_csv(FIGURES_DIR / "shap_feature_importance.csv", index=False)

    # ── Waterfall plot for one true positive ─────────────────────────────
    pos_idx = y_all[y_all == 1].index.tolist()
    if pos_idx:
        case_row = X_proc.iloc[pos_idx[0]]
        case_prob = float(model.predict_proba(case_row.values.reshape(1, -1))[0, 1])
        logger.info("Waterfall case: row %d  prob=%.4f", pos_idx[0], case_prob)

        sv = engine._compute_shap_values(case_row.to_frame().T)
        sv1d = sv[0] if sv.ndim == 2 else sv[0, :, 1]
        exp_val = engine._explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            exp_val = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])

        explanation = shap.Explanation(
            values=sv1d,
            base_values=exp_val,
            data=case_row.values,
            feature_names=feature_cols,
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"SHAP Waterfall — True Positive Case (P={case_prob:.3f})", pad=12)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "shap_waterfall_case.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Waterfall plot saved.")

        # Structured local explanation
        local_exp = engine.explain_instance(case_row, top_k=5)
        print("\n── Local explanation (true positive) ──")
        print(f"  Risk probability : {local_exp['risk_probability']}")
        print(f"  Risk band        : {local_exp['risk_band']} — {local_exp['risk_band_label']}")
        print(f"  Recommended action: {local_exp['recommended_action']}")
        print("  Top factors:")
        for f in local_exp["top_factors"]:
            print(f"    {f['lay_label']:<45} shap={f['shap_value']:+.4f}  {f['direction']} ({f['strength']})")

    # ── Surrogate tree ────────────────────────────────────────────────────
    logger.info("Fitting surrogate decision tree …")
    threshold = 0.438  # v4 classification threshold from model_registry
    y_pred = (model.predict_proba(X_proc.values)[:, 1] >= threshold).astype(int)
    rules = engine.surrogate_tree(X_proc, y_pred, max_depth=4)
    print("\n── Surrogate tree rules (depth=4) ──")
    print(rules[:1500])

    # ── SHAP CSV export ───────────────────────────────────────────────────
    logger.info("Exporting SHAP values CSV …")
    engine.export_shap_csv(X_proc, FIGURES_DIR / "shap_values.csv")

    # ── Fairness audit ────────────────────────────────────────────────────
    logger.info("Running fairness audit …")
    evaluator = ModelEvaluator(cfg)

    threshold_v4 = 0.438
    y_prob = model.predict_proba(X_proc.values)[:, 1]
    y_pred_bin = (y_prob >= threshold_v4).astype(int)

    audit_df_rows = []
    # Gender
    for val in ["male", "female"]:
        mask = df_raw["gender"] == val
        if mask.sum() < 10:
            continue
        tp = int(((y_pred_bin[mask] == 1) & (y_all[mask] == 1)).sum())
        fn = int(((y_pred_bin[mask] == 0) & (y_all[mask] == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        audit_df_rows.append({"group": "gender", "value": val, "n": int(mask.sum()),
                               "n_positive": int(y_all[mask].sum()), "recall": round(recall, 4)})

    # Age group
    age = df_raw["age_years"]
    for label, mask in [("11-13", age <= 13), ("14-16", (age >= 14) & (age <= 16)), ("17-18", age >= 17)]:
        if mask.sum() < 10:
            continue
        tp = int(((y_pred_bin[mask] == 1) & (y_all[mask] == 1)).sum())
        fn = int(((y_pred_bin[mask] == 0) & (y_all[mask] == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        audit_df_rows.append({"group": "age_group", "value": label, "n": int(mask.sum()),
                               "n_positive": int(y_all[mask].sum()), "recall": round(recall, 4)})

    # SEN status (non-missing only)
    sen = df_raw["sen_status"]
    for val, label in [(0, "no_SEN"), (1, "SEN")]:
        mask = sen == val
        if mask.sum() < 10:
            continue
        tp = int(((y_pred_bin[mask] == 1) & (y_all[mask] == 1)).sum())
        fn = int(((y_pred_bin[mask] == 0) & (y_all[mask] == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        audit_df_rows.append({"group": "sen_status", "value": label, "n": int(mask.sum()),
                               "n_positive": int(y_all[mask].sum()), "recall": round(recall, 4)})

    # Deprivation (quintile groups)
    for q_label, mask in [("Q1-2 (deprived)", df_raw["deprivation_quintile"] <= 2),
                           ("Q3 (middle)", df_raw["deprivation_quintile"] == 3),
                           ("Q4-5 (affluent)", df_raw["deprivation_quintile"] >= 4)]:
        if mask.sum() < 10:
            continue
        tp = int(((y_pred_bin[mask] == 1) & (y_all[mask] == 1)).sum())
        fn = int(((y_pred_bin[mask] == 0) & (y_all[mask] == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        audit_df_rows.append({"group": "deprivation", "value": q_label, "n": int(mask.sum()),
                               "n_positive": int(y_all[mask].sum()), "recall": round(recall, 4)})

    audit_df = pd.DataFrame(audit_df_rows)
    audit_df.to_csv(FIGURES_DIR / "fairness_audit.csv", index=False)
    print("\n── Fairness Audit ──")
    print(audit_df.to_string(index=False))

    # Parity check: flag if any group recall differs by >0.10
    recalls = audit_df["recall"].dropna()
    if len(recalls) >= 2:
        gap = recalls.max() - recalls.min()
        flag = "⚠ PARITY GAP > 0.10" if gap > 0.10 else "✓ Parity within 0.10"
        print(f"\n  Recall range: {recalls.min():.3f}–{recalls.max():.3f}  gap={gap:.3f}  {flag}")

    logger.info("All outputs saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
