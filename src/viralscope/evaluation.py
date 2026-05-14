"""Model evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

from .config import TARGET_COLUMN


def _base_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    metrics = {
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "predicted_positive_rate": float(np.mean(y_pred)) if len(y_pred) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
        metrics["brier_score"] = float(brier_score_loss(y_true, np.clip(y_score, 0, 1)))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["brier_score"] = None

    return metrics


def calibration_curve_points(y_true: pd.Series, y_score: np.ndarray, n_bins: int = 10) -> list[dict]:
    scores = np.clip(np.asarray(y_score, dtype=float), 0, 1)
    labels = np.asarray(y_true, dtype=int)
    if len(scores) == 0:
        return []

    bin_edges = np.linspace(0, 1, n_bins + 1)
    assignments = np.digitize(scores, bin_edges[1:-1], right=True)
    rows: list[dict] = []

    for index in range(n_bins):
        mask = assignments == index
        if not np.any(mask):
            continue
        rows.append(
            {
                "bucket": f"{int(bin_edges[index] * 100)}-{int(bin_edges[index + 1] * 100)}",
                "mean_predicted_probability": float(scores[mask].mean()),
                "observed_rate": float(labels[mask].mean()),
                "count": int(mask.sum()),
            }
        )

    return rows


def _downsample_curve(points: list[dict], max_points: int = 60) -> list[dict]:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
    unique_indices = sorted(set(indices.tolist()))
    return [points[index] for index in unique_indices]


def precision_recall_curve_points(
    y_true: pd.Series,
    y_score: np.ndarray,
    max_points: int = 60,
) -> list[dict]:
    if pd.Series(y_true).nunique() < 2:
        return []

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    rows = []
    for index, (precision_value, recall_value) in enumerate(zip(precision, recall, strict=False)):
        threshold = None
        if index < len(thresholds):
            threshold = float(thresholds[index])
        rows.append(
            {
                "precision": float(precision_value),
                "recall": float(recall_value),
                "threshold": threshold,
            }
        )

    ordered = sorted(rows, key=lambda item: item["recall"])
    return _downsample_curve(ordered, max_points=max_points)


def grouped_slice_metrics(evaluation_frame: pd.DataFrame, group_column: str) -> list[dict]:
    if evaluation_frame.empty or group_column not in evaluation_frame.columns:
        return []

    rows = []
    for raw_group, subset in evaluation_frame.groupby(group_column, dropna=False):
        group_name = "Unknown" if pd.isna(raw_group) else str(raw_group)
        y_true = subset[TARGET_COLUMN].astype(int)
        y_pred = subset["predicted_label"].astype(int).to_numpy()
        y_score = subset["predicted_probability"].astype(float).to_numpy()
        metrics = _base_metrics(y_true, y_pred, y_score)
        rows.append({group_column: group_name, **metrics})

    return sorted(rows, key=lambda item: (-item["rows"], str(item[group_column])))


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return portfolio-friendly metrics for an imbalanced classifier."""
    y_true = pd.Series(y_test).astype(int)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    metrics = _base_metrics(y_true, y_pred, y_score)
    metrics["calibration_curve"] = calibration_curve_points(y_true, y_score)
    metrics["precision_recall_curve"] = precision_recall_curve_points(y_true, y_score)

    evaluation_frame = X_test.copy()
    evaluation_frame[TARGET_COLUMN] = y_true.values
    evaluation_frame["predicted_probability"] = np.clip(y_score, 0, 1)
    evaluation_frame["predicted_label"] = y_pred
    metrics["slice_metrics"] = {
        "platform": grouped_slice_metrics(evaluation_frame, "platform"),
        "niche": grouped_slice_metrics(evaluation_frame, "niche"),
    }
    return metrics


def metrics_frame(metrics: dict) -> pd.DataFrame:
    """Convert scalar metrics to a display-friendly dataframe."""
    scalar = {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrix" and isinstance(value, (int, float)) and value is not None
    }
    return pd.DataFrame({"metric": list(scalar.keys()), "value": list(scalar.values())})
