"""End-to-end training pipeline runner.

Usage
-----
python scripts/run_pipeline.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import Config
from src.train import run_training_pipeline

if __name__ == "__main__":
    cfg = Config()
    print(f"\nDataset  : {cfg.active_dataset_key}  ({cfg.active_dataset_path})")
    print(f"Schema   : v{cfg.schema_version}")
    print(f"Outer CV : {cfg.cv_outer_folds}-fold  |  Inner CV : {cfg.cv_inner_folds}-fold")
    print(f"Optuna   : {cfg.optuna_n_trials} trials per model per outer fold")
    print(f"Models dir: models/\n")

    results = run_training_pipeline(cfg)

    print("\n=== FINAL RESULTS ===")
    print(f"Best model : {results['best_model_name']}")
    print(f"Threshold  : {results['threshold']:.4f}")
    print("\nHeld-out test metrics:")
    for k, v in results["test_metrics"].items():
        print(f"  {k:<30} {v:.4f}")
    print("\nArtefacts saved:")
    for name, path in results["artefact_paths"].items():
        print(f"  {name:<16} {path}")
