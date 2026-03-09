"""
Unit tests for the TimeSeriesFeatureGenerator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_generator import TimeSeriesFeatureGenerator


class TestLagFeatures:
    """Tests for lag feature generation."""

    def test_lag_columns_created(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(lag_orders=[1, 7])
        result = gen.generate_lag_features(sample_daily_df)

        assert "lag_1" in result.columns
        assert "lag_7" in result.columns

    def test_lag_values_shifted_correctly(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(lag_orders=[1])
        result = gen.generate_lag_features(sample_daily_df)

        # The lag-1 column at row index 1 should equal the value at row index 0
        expected = sample_daily_df["value"].iloc[0]
        actual = result["lag_1"].iloc[1]
        assert actual == pytest.approx(expected)

    def test_lag_first_row_is_nan(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(lag_orders=[1])
        result = gen.generate_lag_features(sample_daily_df)
        assert np.isnan(result["lag_1"].iloc[0])


class TestRollingFeatures:
    """Tests for rolling-window statistics."""

    def test_rolling_columns_exist(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(rolling_windows=[7])
        result = gen.generate_rolling_features(sample_daily_df)

        for stat in ["mean", "std", "min", "max"]:
            assert f"rolling_{stat}_7" in result.columns

    def test_rolling_mean_reasonable(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(rolling_windows=[7])
        result = gen.generate_rolling_features(sample_daily_df)

        # After a full window the rolling mean should be close to the local average
        rolling_vals = result["rolling_mean_7"].dropna()
        assert rolling_vals.min() > 0, "Rolling mean should be positive for positive data"
        assert rolling_vals.max() < sample_daily_df["value"].max() * 2

    def test_rolling_min_leq_max(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(rolling_windows=[14])
        result = gen.generate_rolling_features(sample_daily_df)
        valid = result.dropna(subset=["rolling_min_14", "rolling_max_14"])
        assert (valid["rolling_min_14"] <= valid["rolling_max_14"]).all()


class TestCalendarFeatures:
    """Tests for calendar-based features."""

    def test_calendar_columns(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator()
        result = gen.generate_calendar_features(sample_daily_df)

        expected_cols = [
            "day_of_week", "day_of_month", "month", "quarter",
            "week_of_year", "is_weekend", "is_month_start", "is_month_end",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing calendar feature: {col}"

    def test_is_weekend_binary(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator()
        result = gen.generate_calendar_features(sample_daily_df)
        unique_vals = set(result["is_weekend"].unique())
        assert unique_vals.issubset({0, 1})

    def test_quarter_range(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator()
        result = gen.generate_calendar_features(sample_daily_df)
        assert result["quarter"].min() >= 1
        assert result["quarter"].max() <= 4


class TestFourierFeatures:
    """Tests for Fourier sine/cosine features."""

    def test_fourier_column_count(self, sample_daily_df: pd.DataFrame) -> None:
        order = 3
        gen = TimeSeriesFeatureGenerator(fourier_order=order)
        result = gen.generate_fourier_features(sample_daily_df)

        sin_cols = [c for c in result.columns if c.startswith("sin_")]
        cos_cols = [c for c in result.columns if c.startswith("cos_")]
        assert len(sin_cols) == order
        assert len(cos_cols) == order

    def test_fourier_values_bounded(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(fourier_order=5)
        result = gen.generate_fourier_features(sample_daily_df)

        for col in result.columns:
            if col.startswith("sin_") or col.startswith("cos_"):
                assert result[col].min() >= -1.0001
                assert result[col].max() <= 1.0001


class TestAllFeatures:
    """Tests for the combined feature generation."""

    def test_all_features_no_nans(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(
            lag_orders=[1, 7],
            rolling_windows=[7],
            fourier_order=2,
        )
        result = gen.generate_all_features(sample_daily_df, drop_na=True)
        assert result.isna().sum().sum() == 0

    def test_all_features_row_count(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(lag_orders=[1, 7])
        result = gen.generate_all_features(sample_daily_df, drop_na=True)
        # Rows lost should be roughly equal to max lag
        assert len(result) < len(sample_daily_df)
        assert len(result) >= len(sample_daily_df) - 30

    def test_get_feature_columns(self, sample_daily_df: pd.DataFrame) -> None:
        gen = TimeSeriesFeatureGenerator(lag_orders=[1], rolling_windows=[7])
        result = gen.generate_all_features(sample_daily_df, drop_na=True)
        feat_cols = gen.get_feature_columns(result)

        assert "lag_1" in feat_cols
        assert "rolling_mean_7" in feat_cols
        assert "timestamp" not in feat_cols
        assert "value" not in feat_cols
