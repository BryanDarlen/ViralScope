"""Prediction interface for trained ViralScope models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import MODEL_ARTIFACT_PATH
from .explanations import explain_prediction, reasoning_summary, recommendations_from_factors
from .training import load_model_artifact


@dataclass
class PredictionResult:
    probability: float
    score: int
    bucket: str
    positive_factors: list[dict]
    negative_factors: list[dict]
    recommendations: list[str]
    reasoning_summary: str
    engineered_features: dict


def probability_bucket(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.55:
        return "Medium-high"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def predict_virality(model, post_data: dict | pd.DataFrame, metadata: dict | None = None) -> PredictionResult:
    """Predict virality probability for one post-like record."""
    if isinstance(post_data, pd.DataFrame):
        if len(post_data) != 1:
            raise ValueError("predict_virality expects exactly one row.")
        raw = post_data.iloc[0].to_dict()
        frame = post_data
    else:
        raw = dict(post_data)
        frame = pd.DataFrame([raw])

    probability = float(model.predict_proba(frame)[0, 1])
    positives, negatives, engineered = explain_prediction(model, raw, metadata=metadata)
    recommendations = recommendations_from_factors(negatives, engineered)
    return PredictionResult(
        probability=probability,
        score=int(round(probability * 100)),
        bucket=probability_bucket(probability),
        positive_factors=positives,
        negative_factors=negatives,
        recommendations=recommendations,
        reasoning_summary=reasoning_summary(probability, positives, negatives),
        engineered_features=engineered,
    )


def predict_from_saved_artifact(
    post_data: dict | pd.DataFrame,
    path: str | Path = MODEL_ARTIFACT_PATH,
) -> PredictionResult:
    model, metadata = load_model_artifact(path)
    return predict_virality(model, post_data, metadata=metadata)
