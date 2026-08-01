"""
Explainability engine — SHAP-based global and local explanations plus a
surrogate decision tree for practitioner-facing output.

Design principles (from ml-development-plan.md Section 5):
* SHAP explanations are mandatory for deployment — no model is deployed
  without them.
* All feature labels are resolved from schema.json via ``Config.get_lay_label()``
  so educator-facing text is never hard-coded.
* The ``explain_instance`` method returns a structured dict suitable for the
  web application's JSON response (ml-development-plan.md Section 8.2).
* SHAP interaction values and global importance plots are saved to disk for
  inclusion in the UKDA model card.

Usage
-----
>>> from src.config import Config
>>> from src.explainability import ExplainabilityEngine
>>> cfg = Config()
>>> engine = ExplainabilityEngine(cfg, model=best_model, feature_names=feature_names)
>>> global_imp = engine.global_shap(X_test)
>>> explanation = engine.explain_instance(X_test.iloc[0])
>>> surrogate_rules = engine.surrogate_tree(X_test, best_model.predict(X_test.values))
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import Config

logger = logging.getLogger(__name__)

# Strength labels mapped from normalised |SHAP value| quartiles
_STRENGTH_LABELS = {3: "strong", 2: "moderate", 1: "mild", 0: "negligible"}


class ExplainabilityEngine:
    """Compute SHAP-based explanations for the fitted classifier.

    Supports ``TreeExplainer`` for tree-based models (Random Forest, XGBoost,
    Decision Tree) and falls back to ``PermutationExplainer`` for linear or
    kernel-based models (Logistic Regression, SVM, MLP).

    Parameters
    ----------
    cfg : Config
        Loaded configuration object (provides lay labels).
    model : fitted sklearn / XGBoost classifier
        The selected model from the training pipeline.
    feature_names : list[str]
        Column names of the preprocessed feature matrix passed to the model.
    figures_dir : Path, optional
        Directory for saving SHAP plots.
    """

    def __init__(
        self,
        cfg: Config,
        model,
        feature_names: list[str],
        figures_dir: Path = Path("reports/figures"),
        background_data: np.ndarray | None = None,
    ) -> None:
        self._cfg = cfg
        self._model = model
        self._feature_names = feature_names
        self._figures_dir = Path(figures_dir)
        self._figures_dir.mkdir(parents=True, exist_ok=True)
        self._background_data = background_data

        self._explainer = self._build_explainer()
        self._global_shap_values: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Explainer construction                                               #
    # ------------------------------------------------------------------ #

    def _build_explainer(self):
        """Build the appropriate SHAP explainer for the model type."""
        model_type = type(self._model).__name__
        tree_types = {
            "RandomForestClassifier",
            "DecisionTreeClassifier",
            "XGBClassifier",
            "GradientBoostingClassifier",
        }
        if model_type in tree_types:
            logger.info("Using shap.TreeExplainer for %s", model_type)
            return shap.TreeExplainer(self._model)
        else:
            logger.info(
                "Using shap.LinearExplainer / PermutationExplainer for %s", model_type
            )
            try:
                if self._background_data is not None:
                    return shap.LinearExplainer(self._model, self._background_data)
                # Without background data masker fails — falls back to KernelExplainer
                return shap.LinearExplainer(self._model, masker=shap.maskers.Independent)
            except Exception:
                return None  # Fall back to per-instance KernelExplainer in explain_instance

    # ------------------------------------------------------------------ #
    # Global explanations                                                  #
    # ------------------------------------------------------------------ #

    def global_shap(self, X: pd.DataFrame, save_plot: bool = True) -> pd.DataFrame:
        """Compute mean |SHAP value| across all samples and return a ranked table.

        Parameters
        ----------
        X : pd.DataFrame
            Preprocessed feature matrix (test set recommended).
        save_plot : bool
            If True, save a beeswarm + bar importance plot to ``figures_dir``.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``lay_label``, ``mean_abs_shap``.
            Sorted by descending importance.
        """
        shap_vals = self._compute_shap_values(X)
        self._global_shap_values = shap_vals

        # For binary classifiers, take the positive class SHAP values
        if shap_vals.ndim == 3:
            shap_pos = shap_vals[:, :, 1]
        else:
            shap_pos = shap_vals

        mean_abs = np.abs(shap_pos).mean(axis=0)
        importance_df = pd.DataFrame(
            {
                "feature": self._feature_names,
                "lay_label": [self._cfg.get_lay_label(f) for f in self._feature_names],
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        if save_plot:
            self._save_global_plot(shap_pos, X)

        logger.info(
            "Top 5 features by global SHAP importance:\n%s",
            importance_df.head(5).to_string(index=False),
        )
        return importance_df

    # ------------------------------------------------------------------ #
    # Local (per-student) explanations                                     #
    # ------------------------------------------------------------------ #

    def explain_instance(self, x: pd.Series | pd.DataFrame, top_k: int = 3) -> dict:
        """Generate a local explanation for a single student.

        Returns a structured dict suitable for the web application's JSON
        response (see ml-development-plan.md Section 8.2).

        Parameters
        ----------
        x : pd.Series or single-row pd.DataFrame
            Preprocessed feature vector for one student.
        top_k : int
            Number of top contributing features to include.

        Returns
        -------
        dict
            Keys: ``risk_probability``, ``risk_band``, ``top_factors``.
        """
        if isinstance(x, pd.Series):
            x_df = x.to_frame().T.reset_index(drop=True)
        else:
            x_df = x.head(1).reset_index(drop=True)

        # Predict probability
        risk_prob = float(self._model.predict_proba(x_df.values)[0, 1])
        risk_band = self._prob_to_band(risk_prob)

        # Compute local SHAP values
        shap_vals = self._compute_shap_values(x_df)
        if shap_vals.ndim == 3:
            local_shap = shap_vals[0, :, 1]
        else:
            local_shap = shap_vals[0, :]

        # Rank features by absolute SHAP value
        top_idx = np.argsort(np.abs(local_shap))[::-1][:top_k]

        # Normalise for strength label
        max_abs = np.abs(local_shap).max() + 1e-9
        quartile_breaks = [0, 0.25 * max_abs, 0.50 * max_abs, 0.75 * max_abs, max_abs]

        factors = []
        for i in top_idx:
            sv = float(local_shap[i])
            col_name = self._feature_names[i]
            strength_idx = int(np.digitize(abs(sv), quartile_breaks[1:]))
            strength_idx = min(strength_idx, 3)
            factors.append(
                {
                    "feature": col_name,
                    "lay_label": self._cfg.get_lay_label(col_name),
                    "shap_value": round(sv, 4),
                    "direction": "increases risk" if sv > 0 else "decreases risk",
                    "strength": _STRENGTH_LABELS[strength_idx],
                }
            )

        return {
            "risk_probability": round(risk_prob, 4),
            "risk_band": risk_band,
            "risk_band_label": self._band_label(risk_band),
            "top_factors": factors,
            "recommended_action": self._recommended_action(risk_band),
            "model_version": "v1",  # Updated by train.py when model is versioned
        }

    # ------------------------------------------------------------------ #
    # SHAP interaction values                                              #
    # ------------------------------------------------------------------ #

    def shap_interaction_plot(
        self,
        X: pd.DataFrame,
        feature_a: str,
        feature_b: str,
    ) -> None:
        """Save a SHAP dependence plot showing the interaction between two features.

        Parameters
        ----------
        X : pd.DataFrame
        feature_a : str
            Primary feature name (x-axis).
        feature_b : str
            Interaction feature name (colour axis).
        """
        if not isinstance(self._explainer, shap.TreeExplainer):
            logger.warning(
                "SHAP interaction values require TreeExplainer. "
                "Skipping interaction plot for %s × %s.", feature_a, feature_b
            )
            return

        if feature_a not in self._feature_names or feature_b not in self._feature_names:
            logger.warning(
                "Feature(s) not in feature_names: %s, %s", feature_a, feature_b
            )
            return

        interaction_vals = self._explainer.shap_interaction_values(X)
        if interaction_vals.ndim == 4:
            interaction_vals = interaction_vals[:, :, :, 1]

        idx_a = self._feature_names.index(feature_a)
        idx_b = self._feature_names.index(feature_b)

        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            X[feature_a], interaction_vals[:, idx_a, idx_a],
            c=X[feature_b], cmap="RdBu", alpha=0.6, edgecolors="none"
        )
        plt.colorbar(scatter, label=self._cfg.get_lay_label(feature_b))
        ax.set_xlabel(self._cfg.get_lay_label(feature_a))
        ax.set_ylabel(f"SHAP value for {self._cfg.get_lay_label(feature_a)}")
        ax.set_title(
            f"SHAP Interaction: {self._cfg.get_lay_label(feature_a)} × "
            f"{self._cfg.get_lay_label(feature_b)}"
        )
        out_path = self._figures_dir / f"shap_interaction_{feature_a}_x_{feature_b}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP interaction plot saved to %s", out_path)

    # ------------------------------------------------------------------ #
    # Surrogate decision tree                                              #
    # ------------------------------------------------------------------ #

    def surrogate_tree(
        self,
        X: pd.DataFrame,
        y_pred: np.ndarray,
        max_depth: int = 4,
    ) -> str:
        """Fit a shallow decision tree to the model's predictions.

        The resulting rule set provides a human-readable approximation of the
        model's decision logic for inclusion in the practitioner guide and
        policy brief.

        Parameters
        ----------
        X : pd.DataFrame
            Full dataset (training + test).
        y_pred : np.ndarray
            Binary predictions from the trained model.
        max_depth : int
            Maximum depth of the surrogate tree (kept shallow for readability).

        Returns
        -------
        str
            Text representation of the surrogate tree rules.
        """
        surrogate = DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight="balanced",
            random_state=self._cfg.random_seed,
        )
        surrogate.fit(X.values, y_pred)
        fidelity = (surrogate.predict(X.values) == y_pred).mean()
        logger.info(
            "Surrogate tree (depth=%d) fidelity: %.3f", max_depth, fidelity
        )

        rules = export_text(surrogate, feature_names=self._feature_names)
        out_path = self._figures_dir / "surrogate_tree_rules.txt"
        out_path.write_text(rules, encoding="utf-8")
        logger.info("Surrogate tree rules saved to %s", out_path)
        return rules

    # ------------------------------------------------------------------ #
    # Export SHAP values for UKDA deposit                                  #
    # ------------------------------------------------------------------ #

    def export_shap_csv(self, X: pd.DataFrame, output_path: Path) -> None:
        """Save per-sample SHAP values as a CSV for inclusion in UKDA deposit.

        The CSV contains one row per sample with one column per feature
        (SHAP value for the positive class).

        Parameters
        ----------
        X : pd.DataFrame
        output_path : Path
            Output file path.
        """
        shap_vals = self._compute_shap_values(X)
        if shap_vals.ndim == 3:
            shap_pos = shap_vals[:, :, 1]
        else:
            shap_pos = shap_vals

        shap_df = pd.DataFrame(
            shap_pos,
            columns=[f"shap_{f}" for f in self._feature_names],
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(output_path, index=False)
        logger.info("SHAP values exported to %s (%d rows)", output_path, len(shap_df))

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values, falling back to KernelExplainer if needed."""
        if self._explainer is None:
            # Last-resort: slow but model-agnostic
            logger.warning("Using KernelExplainer (slow). Consider a tree-based model.")
            ke = shap.KernelExplainer(
                self._model.predict_proba,
                shap.sample(X.values, 50),
            )
            return ke.shap_values(X.values)
        return self._explainer.shap_values(X.values)

    def _prob_to_band(self, prob: float) -> str:
        """Map a predicted probability to a risk band label."""
        # Thresholds are approximate; refined by clinical validation (Week 7)
        if prob < 0.30:
            return "low"
        elif prob < 0.65:
            return "moderate"
        else:
            return "high"

    def _band_label(self, band: str) -> str:
        labels = {
            "low": "Low social phobia risk — no immediate action required",
            "moderate": "Moderate social phobia risk — follow-up recommended",
            "high": "High social phobia risk — refer to pastoral / SEMH team",
        }
        return labels.get(band, band)

    def _recommended_action(self, band: str) -> str:
        actions = {
            "low": "Continue to monitor. No immediate referral needed.",
            "moderate": (
                "Consider a supportive conversation with the student. "
                "Discuss with the pastoral lead if concerns persist."
            ),
            "high": (
                "Refer to the SEMH team or SENCO. This is a screening indicator, "
                "not a diagnosis. Human review is required before any action."
            ),
        }
        return actions.get(band, "")

    def _save_global_plot(self, shap_pos: np.ndarray, X: pd.DataFrame) -> None:
        """Save a beeswarm and bar importance plot."""
        explanation = shap.Explanation(
            values=shap_pos,
            data=X.values,
            feature_names=self._feature_names,
        )

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.bar(explanation, show=False, ax=ax)
        ax.set_title("Global SHAP Feature Importance")
        bar_path = self._figures_dir / "shap_global_importance.png"
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Beeswarm (summary) plot
        fig2 = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_pos, X, feature_names=self._feature_names, show=False)
        plt.title("SHAP Summary (Beeswarm)")
        bee_path = self._figures_dir / "shap_beeswarm.png"
        fig2.savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)

        logger.info("Global SHAP plots saved to %s", self._figures_dir)
