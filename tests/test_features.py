from __future__ import annotations

import pandas as pd

from viralscope.features import FEATURE_COLUMNS, FeatureEngineer, build_features
from viralscope.synthetic import generate_synthetic_posts


def test_build_features_has_expected_columns_and_no_nulls() -> None:
    data = generate_synthetic_posts(n_rows=80, seed=1)
    features = build_features(data)
    assert list(features.columns) == FEATURE_COLUMNS
    assert not features.isna().any().any()


def test_feature_engineer_learns_platform_references() -> None:
    data = generate_synthetic_posts(n_rows=120, seed=2)
    transformer = FeatureEngineer().fit(data)
    transformed = transformer.transform(data.head(5))
    assert "YouTube" in transformer.platform_reference_
    assert len(transformed) == 5
    assert pd.api.types.is_numeric_dtype(transformed["engagement_rate_vs_platform"])
