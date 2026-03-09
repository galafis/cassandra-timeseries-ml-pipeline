"""
LightGBM-based time series forecaster.

Uses a gradient-boosted tree regressor to produce multi-step
forecasts through recursive (auto-regressive) prediction.  Features
are expected to be pre-computed by the feature generator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseTimeSeriesModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.info("lightgbm not installed; LightGBMForecaster will use sklearn GBR")

try:
    from sklearn.ensemble import GradientBoostingRegressor

    SKLEARN_GBR_AVAILABLE = True
except ImportError:
    SKLEARN_GBR_AVAILABLE = False


class LightGBMForecaster(BaseTimeSeriesModel):
    """
    Tabular ML forecaster powered by LightGBM.

    When LightGBM is not installed the model falls back to
    scikit-learn's ``GradientBoostingRegressor``.

    Parameters
    ----------
    params : dict | None
        LightGBM or GBR hyperparameters.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        default_params: Dict[str, Any] = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "verbose": -1,
        }
        merged = {**default_params, **(params or {})}
        super().__init__(name="LightGBM", params=merged)

        self._model: Any = None
        self._feature_columns: List[str] = []
        self._train_dates: Optional[pd.Series] = None
        self._frequency: Optional[str] = None
        self._use_lgb = LIGHTGBM_AVAILABLE

    def fit(
        self,
        train_df: pd.DataFrame,
        date_column: str = "timestamp",
        target_column: str = "value",
        feature_columns: Optional[list[str]] = None,
    ) -> None:
        """
        Train the gradient boosted model on feature-enriched data.

        Parameters
        ----------
        train_df : DataFrame
            Must contain the date column, target column, and all
            generated feature columns.
        feature_columns : list[str] | None
            Explicit list of feature column names.  If ``None``, every
            numeric column except the target and date is used.
        """
        df = train_df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)

        self._train_dates = df[date_column].copy()
        inferred = pd.infer_freq(df[date_column])
        self._frequency = inferred or "D"

        if feature_columns is not None:
            self._feature_columns = list(feature_columns)
        else:
            exclude = {date_column, target_column, "series_id", "metadata"}
            self._feature_columns = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]

        X = df[self._feature_columns].values
        y = df[target_column].values

        if self._use_lgb:
            self._fit_lightgbm(X, y)
        elif SKLEARN_GBR_AVAILABLE:
            self._fit_sklearn(X, y)
        else:
            raise RuntimeError(
                "Neither lightgbm nor sklearn is installed. "
                "Install at least one to use LightGBMForecaster."
            )

        self._is_fitted = True
        logger.info(
            "LightGBMForecaster fitted on %d rows x %d features (backend=%s)",
            X.shape[0],
            X.shape[1],
            "lightgbm" if self._use_lgb else "sklearn-GBR",
        )

    def _fit_lightgbm(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit with the real LightGBM library."""
        lgb_params = {
            k: v
            for k, v in self.params.items()
            if k
            in {
                "n_estimators",
                "learning_rate",
                "num_leaves",
                "max_depth",
                "verbose",
                "objective",
                "metric",
                "boosting_type",
                "feature_fraction",
                "bagging_fraction",
                "bagging_freq",
                "reg_alpha",
                "reg_lambda",
            }
        }
        lgb_params.setdefault("objective", "regression")
        lgb_params.setdefault("verbose", -1)

        self._model = lgb.LGBMRegressor(**lgb_params)
        self._model.fit(X, y)

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit with sklearn's GradientBoostingRegressor."""
        n_est = self.params.get("n_estimators", 200)
        lr = self.params.get("learning_rate", 0.05)
        max_depth = self.params.get("max_depth", 5)
        if max_depth == -1:
            max_depth = 5

        self._model = GradientBoostingRegressor(
            n_estimators=min(n_est, 300),
            learning_rate=lr,
            max_depth=max_depth,
        )
        self._model.fit(X, y)

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Produce forecasts for the next *horizon* steps.

        Parameters
        ----------
        future_df : DataFrame | None
            Pre-built features for the forecast horizon.  Must contain
            the same feature columns used during training plus a
            ``timestamp`` column.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if future_df is not None and not future_df.empty:
            available = [c for c in self._feature_columns if c in future_df.columns]
            if available:
                X_future = future_df[available].values
                preds = self._model.predict(X_future)[:horizon]
                timestamps = (
                    pd.to_datetime(future_df["timestamp"]).values[:horizon]
                    if "timestamp" in future_df.columns
                    else self._generate_future_dates(horizon)
                )
                return pd.DataFrame(
                    {"timestamp": timestamps[:len(preds)], "forecast": preds}
                )

        # Fallback: predict on last known features (naive repeat)
        logger.warning(
            "No future features provided; generating naive forecast"
        )
        future_dates = self._generate_future_dates(horizon)
        last_pred = self._model.predict(
            np.zeros((1, len(self._feature_columns)))
        )[0]
        return pd.DataFrame(
            {"timestamp": future_dates, "forecast": [last_pred] * horizon}
        )

    def _generate_future_dates(self, horizon: int) -> pd.DatetimeIndex:
        """Build future timestamps from the end of the training set."""
        last_date = self._train_dates.max()
        return pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self._frequency),
            periods=horizon,
            freq=self._frequency,
        )

    @property
    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importances if available."""
        if not self._is_fitted or self._model is None:
            return None

        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            return (
                pd.DataFrame(
                    {"feature": self._feature_columns, "importance": importances}
                )
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        return None
