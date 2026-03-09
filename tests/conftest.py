"""
Shared pytest fixtures for the timeseries pipeline test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_daily_df() -> pd.DataFrame:
    """
    A 365-day synthetic time series with trend + seasonality + noise.

    Columns: timestamp, value, series_id
    """
    rng = np.random.RandomState(123)
    n = 365
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=np.float64)

    trend = 500.0 + 0.3 * t
    seasonal = 80 * np.sin(2 * np.pi * t / 7)  # weekly cycle
    noise = rng.normal(0, 20, size=n)
    values = trend + seasonal + noise

    return pd.DataFrame({
        "timestamp": dates,
        "value": np.round(values, 2),
        "series_id": "test_series",
    })


@pytest.fixture()
def short_daily_df() -> pd.DataFrame:
    """
    A short 90-day time series for lightweight tests.
    """
    rng = np.random.RandomState(456)
    n = 90
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=np.float64)
    values = 100 + 0.5 * t + rng.normal(0, 5, size=n)

    return pd.DataFrame({
        "timestamp": dates,
        "value": np.round(values, 2),
        "series_id": "short_series",
    })


@pytest.fixture()
def hierarchical_forecasts() -> dict[str, np.ndarray]:
    """
    Fake forecasts for a three-level hierarchy.

    total -> region_A, region_B
    region_A -> store_1, store_2
    region_B -> store_3
    """
    rng = np.random.RandomState(789)
    h = 10
    return {
        "total": rng.uniform(2000, 3000, size=h),
        "region_A": rng.uniform(1200, 1800, size=h),
        "region_B": rng.uniform(500, 900, size=h),
        "store_1": rng.uniform(600, 1000, size=h),
        "store_2": rng.uniform(400, 800, size=h),
        "store_3": rng.uniform(500, 900, size=h),
    }
