"""
Prophet / statsmodels-based time series forecaster.

Uses Facebook Prophet when available.  If Prophet is not installed the
module falls back to ``statsmodels.tsa.holtwinters.ExponentialSmoothing``
so the pipeline can still run in lightweight environments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseTimeSeriesModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.info("prophet not installed; falling back to statsmodels Holt-Winters")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ProphetForecaster(BaseTimeSeriesModel):
    """
    Forecaster wrapping Prophet with a statsmodels fallback.

    Parameters
    ----------
    params : dict | None
        Prophet hyperparameters (``changepoint_prior_scale``,
        ``seasonality_prior_scale``, ``seasonality_mode``, etc.).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params: Dict[str, Any] = {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
        }
        merged = {**default_params, **(params or {})}
        super().__init__(name="Prophet", params=merged)

        self._model: Any = None
        self._train_df: Optional[pd.DataFrame] = None
        self._frequency: Optional[str] = None
        self._use_prophet = PROPHET_AVAILABLE

    def fit(
        self,
        train_df: pd.DataFrame,
        date_column: str = "timestamp",
        target_column: str = "value",
        feature_columns: Optional[list[str]] = None,
    ) -> None:
        """Train the Prophet (or Holt-Winters fallback) model."""
        df = train_df[[date_column, target_column]].copy()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)

        inferred_freq = pd.infer_freq(df["ds"])
        self._frequency = inferred_freq or "D"

        self._train_df = df.copy()

        if self._use_prophet:
            self._fit_prophet(df)
        elif STATSMODELS_AVAILABLE:
            self._fit_statsmodels(df)
        else:
            raise RuntimeError(
                "Neither prophet nor statsmodels is installed. "
                "Install at least one to use ProphetForecaster."
            )

        self._is_fitted = True
        logger.info(
            "ProphetForecaster fitted on %d observations (backend=%s)",
            len(df),
            "prophet" if self._use_prophet else "statsmodels",
        )

    def _fit_prophet(self, df: pd.DataFrame) -> None:
        """Fit the real Prophet model."""
        prophet_params = {
            k: v
            for k, v in self.params.items()
            if k
            in {
                "changepoint_prior_scale",
                "seasonality_prior_scale",
                "seasonality_mode",
                "yearly_seasonality",
                "weekly_seasonality",
                "daily_seasonality",
                "interval_width",
            }
        }
        self._model = Prophet(**prophet_params)
        self._model.fit(df)

    def _fit_statsmodels(self, df: pd.DataFrame) -> None:
        """Fit Holt-Winters as a fallback."""
        series = df.set_index("ds")["y"]
        series = series.asfreq(self._frequency, method="ffill")

        seasonal_periods = self._guess_seasonal_periods(self._frequency)

        if len(series) >= 2 * seasonal_periods and seasonal_periods > 1:
            self._model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods,
            ).fit(optimized=True)
        else:
            self._model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal=None,
            ).fit(optimized=True)

    @staticmethod
    def _guess_seasonal_periods(freq: str) -> int:
        """Heuristic for seasonal period based on frequency string."""
        freq_upper = (freq or "D").upper()
        if freq_upper.startswith("H"):
            return 24
        if freq_upper.startswith("D"):
            return 7
        if freq_upper.startswith("W"):
            return 52
        if freq_upper.startswith("M"):
            return 12
        return 7

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate point forecasts for *horizon* steps ahead.

        Returns a DataFrame with ``timestamp`` and ``forecast`` columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if self._use_prophet:
            return self._predict_prophet(horizon)
        return self._predict_statsmodels(horizon)

    def _predict_prophet(self, horizon: int) -> pd.DataFrame:
        """Predict using Prophet."""
        future = self._model.make_future_dataframe(periods=horizon, freq=self._frequency)
        forecast = self._model.predict(future)

        out = forecast[["ds", "yhat"]].tail(horizon).copy()
        out.columns = ["timestamp", "forecast"]
        return out.reset_index(drop=True)

    def _predict_statsmodels(self, horizon: int) -> pd.DataFrame:
        """Predict using the Holt-Winters fallback."""
        forecast_values = self._model.forecast(horizon)

        last_date = self._train_df["ds"].max()
        future_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self._frequency),
            periods=horizon,
            freq=self._frequency,
        )

        return pd.DataFrame(
            {"timestamp": future_dates, "forecast": forecast_values.values}
        )
