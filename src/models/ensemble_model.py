"""
Ensemble forecaster combining multiple time series models.

Supports weighted averaging, simple averaging, and median
combination strategies.  Model weights can be tuned manually or
derived from validation performance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseTimeSeriesModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleForecaster(BaseTimeSeriesModel):
    """
    Combines predictions from multiple :class:`BaseTimeSeriesModel`
    instances using a configurable aggregation strategy.

    Parameters
    ----------
    models : list[BaseTimeSeriesModel]
        Constituent forecasters (should already be fitted or will be
        fitted when :meth:`fit` is called).
    weights : dict[str, float] | None
        Per-model weights keyed by ``model.name``.  If ``None``, all
        models contribute equally.
    method : str
        Aggregation method: ``"weighted_average"`` (default),
        ``"simple_average"``, or ``"median"``.
    """

    def __init__(
        self,
        models: List[BaseTimeSeriesModel],
        weights: Optional[Dict[str, float]] = None,
        method: str = "weighted_average",
    ) -> None:
        super().__init__(name="Ensemble", params={"method": method})
        self.models = models
        self.method = method

        if weights is not None:
            self.weights = weights
        else:
            equal_weight = 1.0 / len(models) if models else 1.0
            self.weights = {m.name: equal_weight for m in models}

        self._normalise_weights()

    def _normalise_weights(self) -> None:
        """Ensure weights sum to 1."""
        total = sum(self.weights.get(m.name, 0.0) for m in self.models)
        if total > 0:
            for m in self.models:
                self.weights[m.name] = self.weights.get(m.name, 0.0) / total

    def fit(
        self,
        train_df: pd.DataFrame,
        date_column: str = "timestamp",
        target_column: str = "value",
        feature_columns: Optional[list[str]] = None,
    ) -> None:
        """
        Fit every constituent model on the training data.

        Models that are already fitted are skipped.
        """
        for model in self.models:
            if not model.is_fitted:
                logger.info("Fitting constituent model '%s'", model.name)
                model.fit(
                    train_df,
                    date_column=date_column,
                    target_column=target_column,
                    feature_columns=feature_columns,
                )

        self._is_fitted = all(m.is_fitted for m in self.models)
        logger.info(
            "EnsembleForecaster ready (%d models, method=%s)",
            len(self.models),
            self.method,
        )

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate an ensembled forecast.

        Each constituent model produces its own forecast, and the
        results are combined according to :attr:`method`.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "All constituent models must be fitted before predicting"
            )

        individual_forecasts: Dict[str, np.ndarray] = {}
        timestamps: Optional[np.ndarray] = None

        for model in self.models:
            try:
                pred_df = model.predict(horizon=horizon, future_df=future_df)
                individual_forecasts[model.name] = pred_df["forecast"].values
                if timestamps is None:
                    timestamps = pred_df["timestamp"].values
            except Exception as exc:
                logger.warning(
                    "Model '%s' prediction failed: %s", model.name, exc
                )

        if not individual_forecasts:
            raise RuntimeError("No constituent model produced a forecast")

        combined = self._combine(individual_forecasts)

        if timestamps is None:
            timestamps = pd.date_range("2024-01-01", periods=len(combined), freq="D")

        return pd.DataFrame({"timestamp": timestamps[:len(combined)], "forecast": combined})

    def _combine(self, forecasts: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate individual forecasts into one array."""
        arrays = list(forecasts.values())

        # Align to shortest forecast if lengths differ
        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]

        stacked = np.column_stack(arrays)

        if self.method == "median":
            return np.median(stacked, axis=1)

        if self.method == "simple_average":
            return np.mean(stacked, axis=1)

        # Weighted average (default)
        weight_vector = np.array(
            [
                self.weights.get(name, 1.0 / len(forecasts))
                for name in forecasts
            ]
        )
        weight_vector = weight_vector / weight_vector.sum()
        return stacked @ weight_vector

    def update_weights_from_errors(
        self,
        validation_errors: Dict[str, float],
    ) -> None:
        """
        Set ensemble weights inversely proportional to validation
        errors (lower error = higher weight).

        Parameters
        ----------
        validation_errors : dict[str, float]
            ``{model_name: rmse_or_mae}`` mapping.
        """
        inverse = {
            name: 1.0 / max(err, 1e-8) for name, err in validation_errors.items()
        }
        total = sum(inverse.values())
        self.weights = {name: val / total for name, val in inverse.items()}
        logger.info("Ensemble weights updated: %s", self.weights)
