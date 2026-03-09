"""
Time series feature engineering module.

Generates lag, rolling, calendar, and Fourier features from raw
time series data, preparing it for machine learning models.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesFeatureGenerator:
    """
    Generates tabular features from a time series DataFrame.

    The input DataFrame must contain at least a ``timestamp`` column
    and a ``value`` column.  An optional ``series_id`` column allows
    multi-series feature generation with per-series lag alignment.

    Parameters
    ----------
    lag_orders : list[int]
        Lag steps for autoregressive features.
    rolling_windows : list[int]
        Window sizes for rolling statistics.
    fourier_order : int
        Number of sine/cosine harmonic pairs.
    date_column : str
        Name of the datetime column.
    value_column : str
        Name of the target value column.
    series_id_column : str
        Name of the series identifier column.
    """

    def __init__(
        self,
        lag_orders: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        fourier_order: int = 5,
        date_column: str = "timestamp",
        value_column: str = "value",
        series_id_column: str = "series_id",
    ) -> None:
        self.lag_orders = lag_orders or [1, 2, 3, 7, 14, 28]
        self.rolling_windows = rolling_windows or [7, 14, 30, 90]
        self.fourier_order = fourier_order
        self.date_column = date_column
        self.value_column = value_column
        self.series_id_column = series_id_column

    # ------------------------------------------------------------------
    # Individual feature groups
    # ------------------------------------------------------------------

    def generate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged versions of the target variable.

        For each lag order *k*, a new column ``lag_k`` is added
        containing the value from *k* time steps ago.
        """
        result = df.copy()
        result = result.sort_values(self.date_column)

        for lag in self.lag_orders:
            col_name = f"lag_{lag}"
            if self.series_id_column in result.columns:
                result[col_name] = result.groupby(self.series_id_column)[
                    self.value_column
                ].shift(lag)
            else:
                result[col_name] = result[self.value_column].shift(lag)

        logger.info("Generated %d lag features", len(self.lag_orders))
        return result

    def generate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling-window statistics of the target variable.

        For each window size *w*, four columns are generated:
        ``rolling_mean_w``, ``rolling_std_w``, ``rolling_min_w``,
        and ``rolling_max_w``.
        """
        result = df.copy()
        result = result.sort_values(self.date_column)

        for window in self.rolling_windows:
            if self.series_id_column in result.columns:
                grouped = result.groupby(self.series_id_column)[self.value_column]
                result[f"rolling_mean_{window}"] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=1).mean()
                )
                result[f"rolling_std_{window}"] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=1).std()
                )
                result[f"rolling_min_{window}"] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=1).min()
                )
                result[f"rolling_max_{window}"] = grouped.transform(
                    lambda s: s.rolling(window, min_periods=1).max()
                )
            else:
                rolling = result[self.value_column].rolling(window, min_periods=1)
                result[f"rolling_mean_{window}"] = rolling.mean()
                result[f"rolling_std_{window}"] = rolling.std()
                result[f"rolling_min_{window}"] = rolling.min()
                result[f"rolling_max_{window}"] = rolling.max()

        logger.info(
            "Generated rolling features for %d window sizes",
            len(self.rolling_windows),
        )
        return result

    def generate_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract calendar-based features from the timestamp.

        Adds ``day_of_week``, ``day_of_month``, ``month``, ``quarter``,
        ``week_of_year``, ``is_weekend``, ``is_month_start``, and
        ``is_month_end``.
        """
        result = df.copy()
        ts = pd.to_datetime(result[self.date_column])

        result["day_of_week"] = ts.dt.dayofweek
        result["day_of_month"] = ts.dt.day
        result["month"] = ts.dt.month
        result["quarter"] = ts.dt.quarter
        result["week_of_year"] = ts.dt.isocalendar().week.astype(int)
        result["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        result["is_month_start"] = ts.dt.is_month_start.astype(int)
        result["is_month_end"] = ts.dt.is_month_end.astype(int)

        logger.info("Generated 8 calendar features")
        return result

    def generate_fourier_features(
        self,
        df: pd.DataFrame,
        period: float = 365.25,
    ) -> pd.DataFrame:
        """
        Create Fourier sine/cosine terms for seasonal modelling.

        Parameters
        ----------
        period : float
            The seasonal period in the same unit as the time index
            (e.g. 365.25 for daily data with annual seasonality).
        """
        result = df.copy()
        ts = pd.to_datetime(result[self.date_column])

        time_ordinal = (ts - ts.min()).dt.total_seconds() / 86_400.0

        for k in range(1, self.fourier_order + 1):
            result[f"sin_{k}"] = np.sin(2 * np.pi * k * time_ordinal / period)
            result[f"cos_{k}"] = np.cos(2 * np.pi * k * time_ordinal / period)

        logger.info(
            "Generated %d Fourier term pairs (period=%.1f)",
            self.fourier_order,
            period,
        )
        return result

    # ------------------------------------------------------------------
    # All-in-one
    # ------------------------------------------------------------------

    def generate_all_features(
        self,
        df: pd.DataFrame,
        fourier_period: float = 365.25,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """
        Generate the full feature set in one call.

        Applies lag, rolling, calendar, and Fourier feature generators
        sequentially and optionally drops rows with NaN values introduced
        by lagging and rolling operations.
        """
        result = self.generate_lag_features(df)
        result = self.generate_rolling_features(result)
        result = self.generate_calendar_features(result)
        result = self.generate_fourier_features(result, period=fourier_period)

        n_before = len(result)
        if drop_na:
            result = result.dropna().reset_index(drop=True)
        n_after = len(result)

        logger.info(
            "Full feature set: %d columns, %d rows (%d dropped for NaN)",
            result.shape[1],
            n_after,
            n_before - n_after,
        )
        return result

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Return the names of columns that are generated features.

        Excludes the original timestamp, value, series_id, and metadata
        columns so the caller can easily slice only the feature matrix.
        """
        exclude = {
            self.date_column,
            self.value_column,
            self.series_id_column,
            "metadata",
        }
        return [c for c in df.columns if c not in exclude]
