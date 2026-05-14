from __future__ import annotations

import io
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from viralscope.api_adapters import ApiAdapterError, YouTubeOfficialApiAdapter
from viralscope.cleaning import ensure_virality_label
from viralscope.config import DEMO_DATA_PATH, MODEL_ARTIFACT_PATH, NICHES, SUPPORTED_PLATFORMS
from viralscope.prediction import predict_virality
from viralscope.synthetic import generate_synthetic_posts
from viralscope.training import compare_models, load_model_artifact, save_model_artifact, train_model


TEMPLATE_DIR = ROOT / "webapp" / "templates"
STATIC_DIR = ROOT / "webapp" / "static"

DEFAULT_DEMO_ROWS = 1800
DEFAULT_MODEL = "gradient_boosting"
DEFAULT_SEED = 42
logger = logging.getLogger("viralscope.web")
PREVIEW_COLUMNS = [
    "platform",
    "niche",
    "caption",
    "media_type",
    "early_views",
    "early_likes",
    "early_comments",
    "early_shares",
    "is_viral",
]


class PredictRequest(BaseModel):
    platform: str
    niche: str = "coding"
    caption: str
    hashtags: str = ""
    media_type: str = "short_video"
    post_time: str
    account_follower_count: int = Field(ge=0)
    account_age_days: int = Field(ge=0)
    early_window_hours: float = Field(gt=0)
    early_views: int = Field(ge=0)
    early_likes: int = Field(ge=0)
    early_comments: int = Field(ge=0)
    early_shares: int = Field(ge=0)


class YouTubeLiveRequest(BaseModel):
    video_url_or_id: str = Field(min_length=3)


class DemoTrainRequest(BaseModel):
    rows: int = Field(default=DEFAULT_DEMO_ROWS, ge=300, le=12000)
    model_name: str = DEFAULT_MODEL
    save_model: bool = False


class CompareRequest(BaseModel):
    rows: int = Field(default=DEFAULT_DEMO_ROWS, ge=300, le=12000)


class RuntimeState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.pipeline = None
        self.metadata: dict[str, Any] = {}
        self.evaluation_frame: pd.DataFrame | None = None
        self.dataset_frame: pd.DataFrame | None = None
        self.source = "uninitialized"
        self.demo_rows = DEFAULT_DEMO_ROWS
        self.demo_seed = DEFAULT_SEED

    def ensure_initialized(self) -> None:
        with self.lock:
            if self.pipeline is not None:
                return
            if MODEL_ARTIFACT_PATH.exists():
                pipeline, metadata = load_model_artifact(MODEL_ARTIFACT_PATH)
                metadata = metadata or {}
                metrics = metadata.get("metrics", {})
                if metadata.get("validation") and all(
                    key in metrics for key in ("calibration_curve", "precision_recall_curve", "slice_metrics")
                ):
                    self.pipeline = pipeline
                    self.metadata = metadata
                    self.dataset_frame = generate_synthetic_posts(n_rows=self.demo_rows, seed=self.demo_seed)
                    self.evaluation_frame = None
                    self.source = "saved_model"
                    return
            self._train_demo_unlocked(rows=self.demo_rows, model_name=DEFAULT_MODEL, save_model=False)

    def _apply_training_result(self, result, dataset_frame: pd.DataFrame, source: str) -> None:
        self.pipeline = result.pipeline
        self.metadata = result.metadata
        self.evaluation_frame = result.evaluation_frame.copy()
        self.dataset_frame = ensure_virality_label(dataset_frame).copy()
        self.source = source

    def _train_demo_unlocked(self, rows: int, model_name: str, save_model: bool) -> None:
        dataset = generate_synthetic_posts(n_rows=rows, seed=self.demo_seed)
        result = train_model(dataset, model_name=model_name, calibrate=True)
        if save_model:
            save_model_artifact(result)
        self.demo_rows = rows
        self._apply_training_result(result, dataset, source="synthetic_demo")

    def train_demo(self, rows: int, model_name: str, save_model: bool) -> None:
        with self.lock:
            self._train_demo_unlocked(rows=rows, model_name=model_name, save_model=save_model)

    def train_uploaded(self, dataset: pd.DataFrame, model_name: str, save_model: bool) -> None:
        with self.lock:
            result = train_model(dataset, model_name=model_name, calibrate=True)
            if save_model:
                save_model_artifact(result)
            self._apply_training_result(result, dataset, source="uploaded_csv")


runtime = RuntimeState()
app = FastAPI(title="ViralScope Web", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def serialize_frame(frame: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    preview = frame.head(limit).copy()
    for column in preview.columns:
        preview[column] = preview[column].map(_serialize_value)
    return preview.to_dict(orient="records")


def dataset_summary(data: pd.DataFrame) -> dict[str, Any]:
    labeled = ensure_virality_label(data)
    engagement_rate = (
        (labeled["early_likes"] + labeled["early_comments"] + labeled["early_shares"])
        / labeled["early_views"].replace(0, np.nan)
    ).fillna(0)
    return {
        "rows": int(len(labeled)),
        "viral_rate": float(labeled["is_viral"].mean()),
        "platform_count": int(labeled["platform"].nunique()),
        "median_early_views": int(labeled["early_views"].median()),
        "median_engagement_rate": float(engagement_rate.median()),
    }


def platform_mix(data: pd.DataFrame) -> list[dict[str, Any]]:
    counts = data["platform"].value_counts().rename_axis("platform").reset_index(name="posts")
    return jsonable_encoder(counts.to_dict(orient="records"))


def virality_by_platform(data: pd.DataFrame) -> list[dict[str, Any]]:
    labeled = ensure_virality_label(data)
    summary = labeled.groupby("platform")["is_viral"].mean().mul(100).reset_index(name="viral_rate")
    return jsonable_encoder(summary.to_dict(orient="records"))


def trend_watchlist(data: pd.DataFrame) -> list[dict[str, Any]]:
    top = (
        data.groupby("niche")["five_day_engagement_rate"]
        .mean()
        .sort_values(ascending=False)
        .head(6)
        .reset_index(name="avg_rate")
    )
    return jsonable_encoder(top.to_dict(orient="records"))


def feature_importance(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    importance = metadata.get("feature_importance", [])
    return jsonable_encoder(importance[:12])


def evaluation_histogram(frame: pd.DataFrame | None, bins: int = 12) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    bin_edges = np.linspace(0, 1, bins + 1)
    rows = []
    for label, label_name in [(0, "Non-viral"), (1, "Viral")]:
        subset = frame[frame["is_viral"] == label]
        counts, _ = np.histogram(subset["predicted_probability"], bins=bin_edges)
        for idx, count in enumerate(counts):
            rows.append(
                {
                    "bucket": f"{int(bin_edges[idx] * 100)}-{int(bin_edges[idx + 1] * 100)}",
                    "count": int(count),
                    "label": label_name,
                }
            )
    return rows


def current_dataset() -> pd.DataFrame:
    runtime.ensure_initialized()
    if runtime.dataset_frame is not None:
        return runtime.dataset_frame.copy()
    return generate_synthetic_posts(n_rows=runtime.demo_rows, seed=runtime.demo_seed)


def model_status_payload() -> dict[str, Any]:
    runtime.ensure_initialized()
    metadata = runtime.metadata or {}
    dataset = current_dataset()
    return {
        "model_name": str(metadata.get("model_name", DEFAULT_MODEL)).replace("_", " ").title(),
        "trained_at_utc": metadata.get("trained_at_utc"),
        "rows": int(metadata.get("rows", len(dataset))),
        "positive_rate": float(metadata.get("positive_rate", dataset["is_viral"].mean())),
        "source": runtime.source,
        "metrics": jsonable_encoder(metadata.get("metrics", {})),
        "validation": jsonable_encoder(metadata.get("validation", {})),
        "feature_importance": feature_importance(metadata),
    }


def dashboard_payload(message: str | None = None) -> dict[str, Any]:
    runtime.ensure_initialized()
    dataset = current_dataset()
    labeled = ensure_virality_label(dataset)
    summary = dataset_summary(labeled)
    metrics = runtime.metadata.get("metrics", {})
    top_rows = (
        runtime.evaluation_frame.sort_values("predicted_probability", ascending=False)[
            ["platform", "niche", "caption", "predicted_probability", "is_viral", "predicted_label"]
        ]
        .head(10)
        if runtime.evaluation_frame is not None
        else pd.DataFrame(columns=["platform", "niche", "caption", "predicted_probability", "is_viral", "predicted_label"])
    )
    return {
        "message": message,
        "config": {
            "supported_platforms": SUPPORTED_PLATFORMS,
            "supported_niches": NICHES,
            "current_demo_rows": runtime.demo_rows,
        },
        "model": model_status_payload(),
        "dataset_summary": summary,
        "platform_mix": platform_mix(labeled),
        "virality_by_platform": virality_by_platform(labeled),
        "trend_watchlist": trend_watchlist(labeled),
        "dataset_preview": serialize_frame(labeled[PREVIEW_COLUMNS], limit=18),
        "evaluation": {
            "metrics_table": metrics,
            "confusion_matrix": jsonable_encoder(metrics.get("confusion_matrix", [])),
            "probability_histogram": evaluation_histogram(runtime.evaluation_frame),
            "calibration_curve": jsonable_encoder(metrics.get("calibration_curve", [])),
            "precision_recall_curve": jsonable_encoder(metrics.get("precision_recall_curve", [])),
            "slice_metrics": jsonable_encoder(metrics.get("slice_metrics", {})),
            "top_rows": serialize_frame(top_rows, limit=10),
        },
    }


def prediction_signals(engineered: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        ("View velocity", min(100.0, float(engineered.get("views_per_hour_vs_platform", 0)) * 35)),
        ("Engagement rate", min(100.0, float(engineered.get("engagement_rate_vs_platform", 0)) * 38)),
        ("Comment velocity", min(100.0, float(engineered.get("comments_per_hour_vs_platform", 0)) * 35)),
        ("Share velocity", min(100.0, float(engineered.get("shares_per_hour_vs_platform", 0)) * 35)),
        ("Audience lift", min(100.0, float(engineered.get("views_per_follower", 0)) * 100)),
    ]
    return [{"label": label, "value": round(value, 1)} for label, value in rows]


def parse_csv_upload(upload: UploadFile) -> pd.DataFrame:
    try:
        content = upload.file.read()
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        return pd.read_csv(io.StringIO(text))
    finally:
        upload.file.close()


def live_prediction_payload(post_row: pd.Series) -> dict[str, Any]:
    runtime.ensure_initialized()
    record = post_row.to_dict()
    result = predict_virality(runtime.pipeline, record, metadata=runtime.metadata)
    prediction = asdict(result)
    prediction["signals"] = prediction_signals(result.engineered_features)
    prediction["engineered_features"] = {
        key: _serialize_value(value) for key, value in prediction["engineered_features"].items()
    }
    return {
        "post": {key: _serialize_value(value) for key, value in record.items()},
        "prediction": jsonable_encoder(prediction),
    }


def youtube_error_status(exc: ApiAdapterError) -> int:
    detail = str(exc)
    if "Missing YouTube API key" in detail:
        return 503
    if "Could not extract a YouTube video ID" in detail:
        return 422
    if "YouTube API HTTP" in detail or "YouTube API error" in detail or "YouTube API request failed" in detail:
        return 502
    return 400


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/bootstrap")
def bootstrap() -> dict[str, Any]:
    return dashboard_payload()


@app.get("/api/demo-data.csv")
def demo_csv(rows: int = DEFAULT_DEMO_ROWS) -> StreamingResponse:
    dataset = generate_synthetic_posts(n_rows=max(300, min(rows, 12000)), seed=DEFAULT_SEED)
    csv_text = dataset.to_csv(index=False)
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="viralscope_demo_data.csv"'},
    )


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    runtime.ensure_initialized()
    result = predict_virality(runtime.pipeline, request.model_dump(), metadata=runtime.metadata)
    payload = asdict(result)
    payload["signals"] = prediction_signals(result.engineered_features)
    payload["engineered_features"] = {
        key: _serialize_value(value) for key, value in payload["engineered_features"].items()
    }
    return jsonable_encoder(payload)


@app.post("/api/live/youtube/video")
def predict_live_youtube_video(request: YouTubeLiveRequest) -> dict[str, Any]:
    try:
        frame = YouTubeOfficialApiAdapter().fetch_video(request.video_url_or_id)
    except ApiAdapterError as exc:
        status_code = youtube_error_status(exc)
        logger.warning(
            "Live YouTube lookup failed: status=%s api_key_present=%s input=%r detail=%s",
            status_code,
            bool(os.getenv("YOUTUBE_API_KEY")),
            request.video_url_or_id,
            str(exc),
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    if frame.empty:
        raise HTTPException(status_code=404, detail="No YouTube video data was returned for that URL or ID.")

    return live_prediction_payload(frame.iloc[0])


@app.post("/api/train/demo")
def train_demo_endpoint(request: DemoTrainRequest) -> dict[str, Any]:
    runtime.train_demo(rows=request.rows, model_name=request.model_name, save_model=request.save_model)
    return dashboard_payload(message=f"Demo model retrained on {request.rows:,} synthetic rows.")


@app.post("/api/train/upload")
def train_upload_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL),
    save_model: bool = Form(False),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")
    try:
        dataset = parse_csv_upload(file)
        runtime.train_uploaded(dataset=dataset, model_name=model_name, save_model=save_model)
    except Exception as exc:  # pragma: no cover - keeps API errors readable for users
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return dashboard_payload(message=f"Uploaded CSV trained successfully with {model_name}.")


@app.post("/api/compare")
def compare_endpoint(request: CompareRequest) -> dict[str, Any]:
    dataset = current_dataset() if runtime.source == "uploaded_csv" else generate_synthetic_posts(request.rows, DEFAULT_SEED)
    comparison = compare_models(dataset)
    return {"comparison": jsonable_encoder(comparison.to_dict(orient="records"))}


@app.get("/api/demo-preview")
def demo_preview(limit: int = 18) -> dict[str, Any]:
    dataset = current_dataset()
    return {
        "preview": serialize_frame(dataset[PREVIEW_COLUMNS], limit=limit),
        "summary": dataset_summary(dataset),
    }
