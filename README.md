# ViralScope

ViralScope is a privacy-aware Python machine learning project that predicts whether a social media post is likely to become viral within 5 days.

It is built as a local-first web app:
- `FastAPI` backend
- custom `HTML/CSS/JS` frontend
- Python ML pipeline for cleaning, features, training, evaluation, and prediction

The project is designed to work without scraping restricted platforms. The current MVP supports:
- synthetic demo data
- uploaded CSV training data
- virality probability prediction
- explanation of key positive and negative factors
- model evaluation and dataset preview
- official-adapter helper scripts for YouTube and TikTok connected accounts

## What ViralScope Predicts

Given post-time or early-post features such as:
- platform
- caption
- hashtags
- account size
- posting time
- media type
- early views, likes, comments, and shares

ViralScope outputs a calibrated virality probability from `0-100%`.

Default virality rule:

> A post is considered viral if its five-day engagement rate is in the top 5% compared with posts from the same platform and niche.

If your dataset already contains `is_viral`, ViralScope uses that label directly. Otherwise, it can create labels from:
- `five_day_engagement_rate`
- or final five-day metrics such as views, likes, comments, and shares

## Stack

```text
CSV or synthetic data
        |
        v
cleaning.py
        |
        v
features.py
        |
        v
training.py + evaluation.py
        |
        v
prediction.py + explanations.py
        |
        v
FastAPI API + custom web frontend
```

## Project Structure

```text
.
|-- data/
|   |-- raw/
|   `-- processed/
|-- DATASET_CARD.md
|-- MODEL_CARD.md
|-- models/
|-- notebooks/
|   `-- 01_eda.ipynb
|-- scripts/
|   |-- fetch_tiktok_connected.py
|   |-- fetch_youtube_live.py
|   |-- generate_demo_data.py
|   |-- predict_example.py
|   |-- run_web.py
|   `-- train_model.py
|-- src/
|   `-- viralscope/
|       |-- api_adapters.py
|       |-- cleaning.py
|       |-- config.py
|       |-- evaluation.py
|       |-- explanations.py
|       |-- features.py
|       |-- prediction.py
|       |-- synthetic.py
|       `-- training.py
|-- tests/
|   |-- test_features.py
|   |-- test_training_prediction.py
|   `-- test_web_api.py
|-- webapp/
|   |-- main.py
|   |-- static/
|   |   |-- css/
|   |   `-- js/
|   `-- templates/
|-- privacy.md
|-- pytest.ini
`-- requirements.txt
```

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Start the dashboard

```powershell
python scripts\run_web.py
```

### 4. Open the app

```text
http://127.0.0.1:8000
```

If port `8000` is already in use, `run_web.py` will automatically choose the next open port and print the correct URL.

## Step-by-Step Run Guide

If you want the full local workflow, use this order:

### Step 1. Start the web app
```powershell
python scripts\run_web.py
```

### Step 2. Open the browser

Open the URL printed in the terminal, usually:

```text
http://127.0.0.1:8000
```

### Step 3. Use the dashboard

Inside the app you can:
- run a virality prediction from manual inputs
- retrain the model on synthetic demo data
- upload a CSV and train on your own dataset
- inspect model health and preview data

## Optional CLI Commands

Generate synthetic demo data:

```powershell
python scripts\generate_demo_data.py --rows 2500
```

Train a model from synthetic data:

```powershell
python scripts\train_model.py --model gradient_boosting
```

Run an example prediction:

```powershell
python scripts\predict_example.py
```

Fetch live YouTube data with the official API:

```powershell
python scripts\fetch_youtube_live.py --query "ai productivity" --limit 10
python scripts\fetch_youtube_live.py --video https://www.youtube.com/watch?v=dQw4w9WgXcQ
python scripts\fetch_youtube_live.py --benchmark --limit 20 --output data\raw\youtube_benchmark.csv
```

Fetch TikTok connected-account data:

```powershell
python scripts\fetch_tiktok_connected.py auth-url --client-key YOUR_CLIENT_KEY --redirect-uri https://example.com/callback --state demo-state
python scripts\fetch_tiktok_connected.py profile
python scripts\fetch_tiktok_connected.py videos --limit 10 --output data\raw\tiktok_connected_videos.csv
```

Run tests:

```powershell
pytest
```

## CSV Schema

Recommended input columns:

```text
platform,niche,caption,hashtags,media_type,post_time,
account_follower_count,account_age_days,early_window_hours,
early_views,early_likes,early_comments,early_shares
```

For training, include one of:
- `is_viral`
- `five_day_engagement_rate`
- `five_day_views`, `five_day_likes`, `five_day_comments`, `five_day_shares`

Common aliases such as `title`, `description`, `followers`, `subscribers`, `created_at`, `retweets_early`, and `upvotes_early` are normalized automatically.

## Example Prediction Scenarios

High signal example:
- YouTube short video
- `25,000` followers
- posted at `20:00`
- `18,000` views, `2,100` likes, `320` comments, `450` shares after 2 hours
- expected result: high virality probability

Low signal example:
- Reddit text post
- low account or community size
- posted at `03:00`
- `4` upvotes and `0` comments after 2 hours
- expected result: low virality probability

Medium-high signal example:
- TikTok short video
- `3,000` followers
- `9,500` views, `1,400` likes, `180` comments, `300` shares after 1 hour
- expected result: medium-high to high virality probability

## API-Safe Design

The MVP does not scrape restricted platforms.

Future platform integrations should only be added through official APIs and only when their usage terms allow:
- access
- storage
- model training
- redistribution

See [privacy.md](./privacy.md) for the current data handling policy.

## Live Data Helpers

ViralScope now includes two official-adapter helper paths:

- **YouTube official adapter**
  - search public videos by query
  - fetch a single public video by URL or ID
  - pull a `mostPopular` benchmark set
- **TikTok connected-account adapter**
  - build an authorization URL
  - exchange auth codes for tokens
  - fetch connected-account profile metadata
  - fetch the user's public video metadata

These helpers normalize live responses into the same ViralScope schema used by CSV and synthetic data.

### Environment Variables

Set only what you need:

```powershell
$env:YOUTUBE_API_KEY="your-youtube-api-key"
$env:TIKTOK_ACCESS_TOKEN="your-tiktok-user-access-token"
```

For TikTok authorization code exchange, you will also typically need:

```powershell
$env:TIKTOK_CLIENT_KEY="your-client-key"
$env:TIKTOK_CLIENT_SECRET="your-client-secret"
```

### Important Limitations

- The current web dashboard still uses synthetic data and CSV uploads as the primary in-app workflow.
- Live adapter scripts are available for collection and benchmarking, but they are not yet wired into the main frontend screens.
- Live platform APIs often return current cumulative metrics, not exact first-hour snapshots.
- YouTube does not expose public share counts in the adapter path used here.
- TikTok live support is limited to creator-authorized connected-account data in this implementation.

## Project Documentation

See:

- [MODEL_CARD.md](./MODEL_CARD.md)
- [DATASET_CARD.md](./DATASET_CARD.md)

## Current Status

ViralScope currently includes:
- FastAPI backend using the `src/viralscope` ML modules
- custom frontend built with HTML, CSS, and JavaScript
- synthetic-data and CSV-based training flows
- prediction explanations
- model evaluation surfaces
- temporal holdout validation, calibration diagnostics, and slice metrics
- official YouTube and TikTok connected-account adapter helpers
- automated tests for features, training, prediction, and API behavior
