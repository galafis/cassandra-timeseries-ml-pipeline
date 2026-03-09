"""
Abstract base class for all time series forecasting models.

Every concrete model in the pipeline inherits from
``BaseTimeSeriesModel`` so that the trainer and evaluator can treat
them uniformly through the ``fit`` / ``predict`` / ``evaluate`` API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseTimeSeriesModel(ABC):
    """
    Abstract time series model.

    Subclasses must implement :meth:`fit` and :meth:`predict`.
    A default :meth:`evaluate` is provided that computes RMSE, MAE,
    MAPE, SMAPE, R-squared, and directional accuracy.

    Parameters
    ----------
    name : str
        Human-readable model name.
    params : dict
        Model-specific hyperparameters.
    """

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.params = params or {}
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        date_column: str = "timestamp",
        target_column: str = "value",
        feature_columns: Optional[list[str]] = None,
    ) -> None:
        """Train the model on historical data."""

    @abstractmethod
    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Produce forecasts.

        Returns a DataFrame with at least ``timestamp`` and ``forecast``
        columns.  The ``future_df`` argument carries pre-built features
        for ML models that need exogenous inputs.
        """

    def evaluate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute a standard set of regression metrics.

        Parameters
        ----------
        actual : array-like
            Ground-truth values.
        predicted : array-like
            Point forecasts from the model.

        Returns
        -------
        dict
            Metrics dictionary with keys ``rmse``, ``mae``, ``mape``,
            ``smape``, ``r2``, and ``directional_accuracy``.
        """
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)

        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        r2 = float(r2_score(actual, predicted))

        # MAPE -- guard against division by zero
        nonzero_mask = actual != 0
        if nonzero_mask.any():
            mape = float(
                np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100
            )
        else:
            mape = float("inf")

        # Symmetric MAPE
        denom = np.abs(actual) + np.abs(predicted)
        safe_denom = np.where(denom == 0, 1.0, denom)
        smape = float(
            np.mean(2.0 * np.abs(actual - predicted) / safe_denom) * 100
        )

        # Directional accuracy (percentage of correct direction predictions)
        if len(actual) > 1:
            actual_dir = np.diff(actual) > 0
            pred_dir = np.diff(predicted) > 0
            directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)
        else:
            directional_accuracy = 0.0

        metrics = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 4),
            "smape": round(smape, 4),
            "r2": round(r2, 4),
            "directional_accuracy": round(directional_accuracy, 4),
        }

        logger.info("Model '%s' evaluation: %s", self.name, metrics)
        return metrics

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted
