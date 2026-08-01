"""
Evaluation module — comprehensive model assessment including clinical metrics,
calibration, and fairness / bias audit.

Metrics strategy (from ml-development-plan.md Section 7):
* Primary: Recall ≥ 0.85 (hard gate), PR-AUC, F1, AUC-ROC.
* Clinical: Sensitivity, Specificity, PPV, NPV, FNR, NNS.
* Calibration: Brier score, Expected Calibration Error.
* Fairness: Recall parity across gender, key_stage, SEN, FSM, ethnicity.

Usage
-----
>>> from src.config import Config
>>> from src.evaluate import ModelEvaluator
>>> cfg = Config()
>>> evaluator = ModelEvaluator(cfg)
>>> report = evaluator.evaluate(model, X_test, y_test, threshold=0.42)
>>> evaluator.fairness_audit(model, X_test_raw, y_test, threshold=0.42)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import Config

logger = logging.getLogger(__name__)

# Minimum group size for fairness metrics to be reported
_MIN_GROUP_N = 10


@dataclass
class EvaluationReport:
    """Structured evaluation output for a single model / split.

    Attributes
    ----------
    split_label : str
        e.g. ``'validation'`` or ``'test'``.
    threshold : float
        Classification threshold applied to predicted probabilities.
    n_samples : int
        Total number of samples evaluated.
    n_positive : int
        Number of at-risk students in the true labels.
    metrics : dict
        All computed metrics.
    """

    split_label: str
    threshold: float
    n_samples: int
    n_positive: int
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a printable summary table."""
        lines = [
            f"\n{'─' * 55}",
            f"  Evaluation Report — {self.split_label}",
            f"{'─' * 55}",
            f"  Samples    : {self.n_samples}  |  At-risk: {self.n_positive} "
            f"({100*self.n_positive/max(self.n_samples,1):.1f}%)",
            f"  Threshold  : {self.threshold:.4f}",
            f"{'─' * 55}",
        ]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:<30} {v:.4f}")
            else:
                lines.append(f"  {k:<30} {v}")
        lines.append(f"{'─' * 55}")
        return "\n".join(lines)


class ModelEvaluator:
    """Compute all evaluation metrics defined in ml-development-plan.md.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object.
    figures_dir : Path, optional
        Directory for saving diagnostic plots.
    """

    def __init__(
        self, cfg: Config, figures_dir: Path = Path("reports/figures")
    ) -> None:
        self._cfg = cfg
        self._figures_dir = figures_dir
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Primary evaluation                                                   #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5,
        split_label: str = "test",
        save_plots: bool = True,
    ) -> EvaluationReport:
        """Run the full evaluation suite and return an EvaluationReport.

        Parameters
        ----------
        model : fitted classifier with ``predict_proba``.
        X : pd.DataFrame
            Preprocessed feature matrix.
        y : pd.Series
            True binary labels (spin_risk_binary).
        threshold : float
            Classification threshold (from threshold_optimisation step).
        split_label : str
            Human-readable label for the split (e.g. ``'test'``).
        save_plots : bool
            If True, save ROC and PR curve plots to ``figures_dir``.
        """
        y_prob = model.predict_proba(X.values)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        y_np = y.values.astype(int)

        metrics = {}
        metrics.update(self._primary_metrics(y_np, y_pred, y_prob))
        metrics.update(self._clinical_metrics(y_np, y_pred, y_prob))
        metrics.update(self._calibration_metrics(y_np, y_prob))

        report = EvaluationReport(
            split_label=split_label,
            threshold=threshold,
            n_samples=len(y_np),
            n_positive=int(y_np.sum()),
            metrics=metrics,
        )
        logger.info("%s", report.summary())

        if save_plots:
            self._save_plots(model, X, y_np, y_prob, split_label)

        return report

    # ------------------------------------------------------------------ #
    # Fairness audit                                                       #
    # ------------------------------------------------------------------ #

    def fairness_audit(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        df_raw: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Compute evaluation metrics stratified by fairness-audit columns.

        Implements the bias audit from ml-development-plan.md Section 7.4.
        Groups with fewer than ``_MIN_GROUP_N`` records are suppressed.

        Parameters
        ----------
        model : fitted classifier.
        X : pd.DataFrame
            Preprocessed features (same index as y and df_raw).
        y : pd.Series
            True labels.
        df_raw : pd.DataFrame
            Original (un-preprocessed) DataFrame containing fairness columns.
        threshold : float

        Returns
        -------
        pd.DataFrame
            Rows = fairness subgroups; columns = key metrics.
        """
        audit_cols = [
            c for c in self._cfg.fairness_audit_cols if c in df_raw.columns
        ]
        if not audit_cols:
            logger.warning("No fairness audit columns found in df_raw. Skipping.")
            return pd.DataFrame()

        y_prob = model.predict_proba(X.values)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        y_np = y.values.astype(int)

        rows = []
        for col in audit_cols:
            groups = df_raw[col].fillna("unknown")
            for group_val, group_idx in groups.groupby(groups).groups.items():
                if len(group_idx) < _MIN_GROUP_N:
                    logger.debug(
                        "Fairness audit: suppressed %s=%s (n=%d < %d)",
                        col, group_val, len(group_idx), _MIN_GROUP_N,
                    )
                    continue
                g_y = y_np[group_idx]
                g_pred = y_pred[group_idx]
                g_prob = y_prob[group_idx]

                row = {
                    "audit_col": col,
                    "group_value": str(group_val),
                    "n": len(g_y),
                    "n_positive": int(g_y.sum()),
                    "prevalence_pct": round(100 * g_y.mean(), 1),
                    "recall": round(recall_score(g_y, g_pred, zero_division=0), 4),
                    "precision": round(precision_score(g_y, g_pred, zero_division=0), 4),
                    "f1": round(f1_score(g_y, g_pred, zero_division=0), 4),
                }
                if g_y.sum() > 0 and g_y.sum() < len(g_y):
                    row["auc_roc"] = round(roc_auc_score(g_y, g_prob), 4)
                else:
                    row["auc_roc"] = None  # Cannot compute AUC with single class
                rows.append(row)

        audit_df = pd.DataFrame(rows)
        self._check_fairness_gates(audit_df)
        return audit_df

    # ------------------------------------------------------------------ #
    # Sensitivity analysis                                                 #
    # ------------------------------------------------------------------ #

    def complete_case_sensitivity(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        df_raw: pd.DataFrame,
        threshold: float,
    ) -> dict[str, EvaluationReport]:
        """Compare metrics on full vs. complete-case (non-imputed) samples.

        A AUC-ROC difference > 0.05 between the two subsets flags a potential
        MICE imputation assumption violation.
        """
        results = {}

        # Full dataset
        results["full"] = self.evaluate(
            model, X, y, threshold=threshold, split_label="full_set", save_plots=False
        )

        # Complete cases only (no imputed SPIN items)
        complete_flag_col = "spin_complete_flag"
        if complete_flag_col not in df_raw.columns:
            logger.warning("'spin_complete_flag' absent — skipping complete-case analysis.")
            return results

        complete_idx = df_raw.index[df_raw[complete_flag_col].fillna(0).astype(int) == 1]
        complete_idx = complete_idx[complete_idx.isin(X.index)]
        if len(complete_idx) < 20:
            logger.warning("Too few complete cases (%d) for sensitivity analysis.", len(complete_idx))
            return results

        results["complete_case"] = self.evaluate(
            model,
            X.loc[complete_idx],
            y.loc[complete_idx],
            threshold=threshold,
            split_label="complete_case",
            save_plots=False,
        )

        auc_diff = abs(
            results["full"].metrics.get("auc_roc", 0)
            - results["complete_case"].metrics.get("auc_roc", 0)
        )
        if auc_diff > 0.05:
            logger.warning(
                "AUC-ROC differs by %.4f between full and complete-case sets. "
                "MICE imputation assumptions may be violated. Document as limitation.",
                auc_diff,
            )
        return results

    # ------------------------------------------------------------------ #
    # Metric implementations                                               #
    # ------------------------------------------------------------------ #

    def _primary_metrics(
        self, y: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> dict:
        return {
            "recall": recall_score(y, y_pred, zero_division=0),
            "precision": precision_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_prob) if y.sum() > 0 else float("nan"),
            "pr_auc": average_precision_score(y, y_prob) if y.sum() > 0 else float("nan"),
        }

    def _clinical_metrics(
        self, y: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> dict:
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)  # Positive Predictive Value
        npv = tn / max(tn + fn, 1)  # Negative Predictive Value
        fnr = fn / max(fn + tp, 1)  # False Negative Rate

        prevalence = y.mean()
        nns = (1 / max(sensitivity, 1e-9)) * (1 / max(prevalence, 1e-9)) if prevalence > 0 else float("nan")

        return {
            "sensitivity (recall)": sensitivity,
            "specificity": specificity,
            "ppv (precision)": ppv,
            "npv": npv,
            "false_negative_rate": fnr,
            "number_needed_to_screen": round(nns, 1) if not np.isnan(nns) else float("nan"),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

    def _calibration_metrics(
        self, y: np.ndarray, y_prob: np.ndarray
    ) -> dict:
        brier = brier_score_loss(y, y_prob)

        # Expected Calibration Error (binned approximation, 10 bins)
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += (mask.sum() / len(y)) * abs(bin_acc - bin_conf)

        return {
            "brier_score": brier,
            "expected_calibration_error": round(ece, 4),
        }

    # ------------------------------------------------------------------ #
    # Fairness gate check                                                  #
    # ------------------------------------------------------------------ #

    def _check_fairness_gates(self, audit_df: pd.DataFrame) -> None:
        """Warn if any group-level Recall gap exceeds 0.10."""
        for col, group in audit_df.groupby("audit_col"):
            recalls = group["recall"].dropna()
            if len(recalls) < 2:
                continue
            gap = recalls.max() - recalls.min()
            if gap > 0.10:
                logger.warning(
                    "Fairness gate: Recall gap for '%s' = %.3f > 0.10. "
                    "Review model before deployment (may require retraining).",
                    col, gap,
                )

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #

    def _save_plots(
        self,
        model,
        X: pd.DataFrame,
        y: np.ndarray,
        y_prob: np.ndarray,
        split_label: str,
    ) -> None:
        """Save ROC curve, PR curve, and confusion matrix plots."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        RocCurveDisplay.from_predictions(y, y_prob, ax=axes[0])
        axes[0].set_title(f"ROC Curve — {split_label}")

        PrecisionRecallDisplay.from_predictions(y, y_prob, ax=axes[1])
        axes[1].set_title(f"Precision-Recall Curve — {split_label}")

        y_pred_default = (y_prob >= 0.5).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred_default, display_labels=["Not at-risk", "At-risk"], ax=axes[2]
        )
        axes[2].set_title(f"Confusion Matrix (threshold=0.5) — {split_label}")

        fig.tight_layout()
        out_path = self._figures_dir / f"evaluation_{split_label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Evaluation plots saved to %s", out_path)
