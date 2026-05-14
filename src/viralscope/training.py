"""Training pipeline for the ViralScope MVP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import ensure_virality_label
from .config import MODEL_ARTIFACT_PATH, RANDOM_STATE, TARGET_COLUMN
from .evaluation import evaluate_model
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FeatureEngineer, feature_reference_from_transformer


@dataclass
class TrainingResult:
    pipeline: Pipeline
    metrics: dict
    metadata: dict
    evaluation_frame: pd.DataFrame


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_estimator(model_name: str = "gradient_boosting", random_state: int = RANDOM_STATE):
    model_name = model_name.lower().strip()
    if model_name in {"logistic", "logistic_regression", "baseline"}:
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    if model_name in {"random_forest", "rf"}:
        return RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name in {"gradient_boosting", "gb"}:
        return GradientBoostingClassifier(random_state=random_state)
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost to use model_name='xgboost'.") from exc
        return XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("Install lightgbm to use model_name='lightgbm'.") from exc
        return LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def _maybe_calibrated(estimator, y: pd.Series, calibrate: bool):
    if not calibrate:
        return estimator
    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() < 3:
        return estimator
    cv = int(min(3, counts.min()))
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)


def _extract_feature_importance(pipeline: Pipeline, top_n: int = 20) -> list[dict]:
    try:
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        model = pipeline.named_steps["model"]
    except Exception:
        return []

    candidates = []
    if hasattr(model, "feature_importances_") or hasattr(model, "coef_"):
        candidates = [model]
    elif hasattr(model, "calibrated_classifiers_"):
        candidates = [item.estimator for item in model.calibrated_classifiers_ if hasattr(item, "estimator")]

    values = []
    for candidate in candidates:
        if hasattr(candidate, "feature_importances_"):
            values.append(np.asarray(candidate.feature_importances_, dtype=float))
        elif hasattr(candidate, "coef_"):
            values.append(np.abs(np.asarray(candidate.coef_, dtype=float)).ravel())

    if not values:
        return []

    importance = np.mean(values, axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    return [
        {"feature": str(names[index]), "importance": float(importance[index])}
        for index in order
        if index < len(names)
    ]


def _build_metadata(
    pipeline: Pipeline,
    data: pd.DataFrame,
    metrics: dict,
    model_name: str,
    random_state: int,
    validation: dict,
) -> dict:
    feature_step = pipeline.named_steps["features"]
    return {
        "project": "ViralScope",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "random_state": random_state,
        "rows": int(len(data)),
        "positive_rate": float(data[TARGET_COLUMN].mean()),
        "target": "Top 5% five-day engagement rate within platform/niche, or supplied is_viral label.",
        "validation": validation,
        "raw_input_columns": [
            "platform",
            "niche",
            "caption",
            "hashtags",
            "media_type",
            "post_time",
            "account_follower_count",
            "account_age_days",
            "early_window_hours",
            "early_views",
            "early_likes",
            "early_comments",
            "early_shares",
        ],
        "metrics": metrics,
        "feature_reference": feature_reference_from_transformer(feature_step),
        "feature_importance": _extract_feature_importance(pipeline),
    }


def _timestamp_or_none(series: pd.Series, fn: str) -> str | None:
    if series.empty:
        return None
    timestamp = getattr(series, fn)()
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).isoformat()


def _split_summary(strategy: str, train: pd.DataFrame, test: pd.DataFrame, note: str | None = None) -> dict:
    summary = {
        "strategy": strategy,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start_utc": _timestamp_or_none(train["post_time"], "min"),
        "train_end_utc": _timestamp_or_none(train["post_time"], "max"),
        "test_start_utc": _timestamp_or_none(test["post_time"], "min"),
        "test_end_utc": _timestamp_or_none(test["post_time"], "max"),
    }
    if note:
        summary["note"] = note
    return summary


def _has_both_classes(values: pd.Series) -> bool:
    return pd.Series(values).nunique() >= 2


def _random_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        _split_summary(
            "random_stratified_holdout" if stratify is not None else "random_holdout",
            X_train,
            X_test,
            note="Fallback used because temporal holdout could not keep both classes in train and test.",
        ),
    )


def _temporal_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    ordered = X.copy()
    ordered["_target"] = y.values
    ordered = ordered.sort_values("post_time", kind="mergesort").reset_index(drop=True)
    total_rows = len(ordered)
    if total_rows < 10:
        return _random_holdout_split(X, y, test_size=test_size, random_state=random_state)

    target_test_rows = max(1, int(round(total_rows * test_size)))
    min_train_rows = max(2, total_rows - max(target_test_rows, total_rows - 2))
    ideal_split = total_rows - target_test_rows
    split_candidates = list(range(ideal_split, min_train_rows - 1, -1))
    split_candidates.extend(range(ideal_split + 1, total_rows - 1))

    for split_idx in split_candidates:
        train = ordered.iloc[:split_idx].copy()
        test = ordered.iloc[split_idx:].copy()
        if train.empty or test.empty:
            continue
        if not _has_both_classes(train["_target"]) or not _has_both_classes(test["_target"]):
            continue

        X_train = train.drop(columns=["_target"])
        X_test = test.drop(columns=["_target"])
        y_train = train["_target"].astype(int)
        y_test = test["_target"].astype(int)
        return X_train, X_test, y_train, y_test, _split_summary("temporal_holdout", X_train, X_test)

    return _random_holdout_split(X, y, test_size=test_size, random_state=random_state)


def train_model(
    df: pd.DataFrame,
    model_name: str = "gradient_boosting",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    calibrate: bool = True,
    validation_strategy: str = "temporal",
) -> TrainingResult:
    """Train a full sklearn Pipeline from raw CSV-style rows."""
    data = ensure_virality_label(df)
    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Training requires at least one viral and one non-viral example.")

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    if validation_strategy.lower().strip() == "temporal":
        X_train, X_test, y_train, y_test, validation = _temporal_holdout_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        X_train, X_test, y_train, y_test, validation = _random_holdout_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    estimator = make_estimator(model_name=model_name, random_state=random_state)
    estimator = _maybe_calibrated(estimator, y_train, calibrate=calibrate)

    pipeline = Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    metadata = _build_metadata(pipeline, data, metrics, model_name, random_state, validation)

    evaluation_frame = X_test.copy()
    evaluation_frame[TARGET_COLUMN] = y_test.values
    evaluation_frame["predicted_probability"] = pipeline.predict_proba(X_test)[:, 1]
    evaluation_frame["predicted_label"] = pipeline.predict(X_test)

    return TrainingResult(
        pipeline=pipeline,
        metrics=metrics,
        metadata=metadata,
        evaluation_frame=evaluation_frame,
    )


def compare_models(
    df: pd.DataFrame,
    model_names: Iterable[str] = ("logistic", "random_forest", "gradient_boosting"),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        result = train_model(df, model_name=model_name, random_state=random_state)
        row = {"model": model_name}
        row.update(
            {
                key: value
                for key, value in result.metrics.items()
                if isinstance(value, (int, float)) or value is None
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["average_precision", "roc_auc", "f1"], ascending=False)


def save_model_artifact(result: TrainingResult, path: str | Path = MODEL_ARTIFACT_PATH) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": result.pipeline, "metadata": result.metadata}, output_path)
    return output_path


def load_model_artifact(path: str | Path = MODEL_ARTIFACT_PATH) -> tuple[Pipeline, dict]:
    artifact = joblib.load(path)
    return artifact["pipeline"], artifact.get("metadata", {})
