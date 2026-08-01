"""
Social Phobia Detection — ML pipeline package.

Entry points
------------
- src.config          : Config (loads schema.json + YAML registries)
- src.data_loader     : load_active_dataset()
- src.schema_validator: SchemaValidator
- src.feature_engineering : FeatureEngineer
- src.preprocessing   : Preprocessor
- src.train           : run_training_pipeline()
- src.evaluate        : ModelEvaluator
- src.explainability  : ExplainabilityEngine
"""

__version__ = "0.1.0"
