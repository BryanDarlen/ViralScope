# ViralScope

Predicting social media content virality across Twitter/X, Reddit, TikTok, and YouTube using machine learning.

## What it does

Given a piece of social media content's features at the time of posting (caption, hashtags, account size, post timing, media type, etc.), ViralScope outputs a probability (0–100%) that the post will go viral within 5 days.

## Locked decisions

**Virality label:** Top 5% per platform on the platform-specific engagement rate, measured at 5 days post-publish.

| Platform | Engagement rate |
|---|---|
| Twitter/X | `(likes + retweets + replies) / followers_at_post_time` |
| Reddit | `upvote_ratio × score / subreddit_median_score` |
| TikTok | `(likes + shares + comments) / views` |
| YouTube | `(likes + comments) / views`; secondary: `views / subscribers` |

**Class imbalance:** ~95/5 by construction. Plan compares SMOTE, class weighting, and focal loss rather than pre-picking.

**Primary model:** XGBoost / LightGBM. Baselines: Logistic Regression, Random Forest.

## Project structure

```
.
├── data/
│   ├── raw/          # Untouched scraped/downloaded data (gitignored)
│   └── processed/    # Cleaned, feature-engineered data (gitignored)
├── notebooks/
│   └── 01_eda.ipynb  # Exploratory analysis
├── src/              # Python modules (collection, features, training, prediction)
├── models/           # Trained .pkl artefacts (gitignored)
├── reports/          # Final 1-page report and figures
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pipeline

Raw collection → Cleaning/merging → EDA → Feature engineering → Training/tuning → Evaluation/SHAP → Streamlit deployment.

## Status

Kickoff. Threshold locked; scaffolding done; data collection next.
