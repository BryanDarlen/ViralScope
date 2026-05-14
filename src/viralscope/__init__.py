"""ViralScope package.

CSV-first, API-safe virality prediction utilities for the portfolio MVP.
"""

from .prediction import PredictionResult, predict_virality
from .training import TrainingResult, load_model_artifact, save_model_artifact, train_model

__all__ = [
    "PredictionResult",
    "TrainingResult",
    "load_model_artifact",
    "predict_virality",
    "save_model_artifact",
    "train_model",
]
