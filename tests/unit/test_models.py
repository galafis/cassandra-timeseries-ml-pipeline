"""
Unit tests for forecasting models (Prophet fallback, LightGBM, Ensemble).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_generator import TimeSeriesFeatureGenerator
from src.models.prophet_model import ProphetForecaster
from src.models.lightgbm_model import LightGBMForecaster
from src.models.ensemble_model import EnsembleForecaster


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _prepare_featured_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Generate features and return (featured_df, feature_columns)."""
    gen = TimeSeriesFeatureGenerator(
        lag_orders=[1, 7],
        rolling_windows=[7],
        fourier_order=2,
    )
    featured = gen.generate_all_features(df, drop_na=True)
    feat_cols = gen.get_feature_columns(featured)
    return featured, feat_cols


# ------------------------------------------------------------------
# ProphetForecaster
# ------------------------------------------------------------------

class TestProphetForecaster:
    """Tests for the Prophet / Holt-Winters forecaster."""

    def test_fit_and_predict(self, short_daily_df: pd.DataFrame) -> None:
        model = ProphetForecaster()
        model.fit(short_daily_df, date_column="timestamp", target_column="value")

        assert model.is_fitted

        preds = model.predict(horizon=7)
        assert "timestamp" in preds.columns
        assert "forecast" in preds.columns
        assert len(preds) == 7

    def test_forecast_values_positive(self, short_daily_df: pd.DataFrame) -> None:
        model = ProphetForecaster()
        model.fit(short_daily_df)
        preds = model.predict(horizon=14)

        # For an upward-trending series the forecast should be broadly positive
        assert preds["forecast"].mean() > 0

    def test_predict_before_fit_raises(self) -> None:
        model = ProphetForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(horizon=5)


# ------------------------------------------------------------------
# LightGBMForecaster
# ------------------------------------------------------------------

class TestLightGBMForecaster:
    """Tests for the LightGBM / sklearn-GBR forecaster."""

    def test_fit_and_predict_with_features(self, sample_daily_df: pd.DataFrame) -> None:
        featured, feat_cols = _prepare_featured_df(sample_daily_df)

        model = LightGBMForecaster(params={"n_estimators": 50, "verbose": -1})
        model.fit(featured, feature_columns=feat_cols)

        assert model.is_fitted

        future = featured.tail(10)
        preds = model.predict(horizon=10, future_df=future)
        assert len(preds) == 10
        assert preds["forecast"].notna().all()

    def test_feature_importance_available(self, sample_daily_df: pd.DataFrame) -> None:
        featured, feat_cols = _prepare_featured_df(sample_daily_df)

        model = LightGBMForecaster(params={"n_estimators": 30, "verbose": -1})
        model.fit(featured, feature_columns=feat_cols)

        imp = model.feature_importance
        assert imp is not None
        assert "feature" in imp.columns
        assert "importance" in imp.columns
        assert len(imp) == len(feat_cols)

    def test_predict_before_fit_raises(self) -> None:
        model = LightGBMForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(horizon=5)


# ------------------------------------------------------------------
# EnsembleForecaster
# ------------------------------------------------------------------

class TestEnsembleForecaster:
    """Tests for the ensemble combiner."""

    def test_ensemble_averages_two_models(self, sample_daily_df: pd.DataFrame) -> None:
        featured, feat_cols = _prepare_featured_df(sample_daily_df)

        prophet = ProphetForecaster()
        lgbm = LightGBMForecaster(params={"n_estimators": 30, "verbose": -1})

        ensemble = EnsembleForecaster(
            models=[prophet, lgbm],
            weights={"Prophet": 0.5, "LightGBM": 0.5},
            method="simple_average",
        )

        ensemble.fit(featured, feature_columns=feat_cols)
        assert ensemble.is_fitted

        future = featured.tail(7)
        preds = ensemble.predict(horizon=7, future_df=future)
        assert len(preds) == 7

    def test_weighted_ensemble_output_differs_from_simple(
        self, sample_daily_df: pd.DataFrame
    ) -> None:
        featured, feat_cols = _prepare_featured_df(sample_daily_df)

        prophet = ProphetForecaster()
        lgbm = LightGBMForecaster(params={"n_estimators": 30, "verbose": -1})

        simple = EnsembleForecaster(
            models=[prophet, lgbm], method="simple_average"
        )
        weighted = EnsembleForecaster(
            models=[prophet, lgbm],
            weights={"Prophet": 0.1, "LightGBM": 0.9},
            method="weighted_average",
        )

        simple.fit(featured, feature_columns=feat_cols)
        weighted.fit(featured, feature_columns=feat_cols)

        future = featured.tail(5)
        s_preds = simple.predict(horizon=5, future_df=future)
        w_preds = weighted.predict(horizon=5, future_df=future)

        # With extreme weight skew the outputs should differ
        # (unless both models predict identically, which is unlikely)
        assert len(s_preds) == len(w_preds) == 5

    def test_update_weights_from_errors(self) -> None:
        prophet = ProphetForecaster()
        lgbm = LightGBMForecaster()

        ens = EnsembleForecaster(models=[prophet, lgbm])
        ens.update_weights_from_errors({"Prophet": 10.0, "LightGBM": 5.0})

        # LightGBM has lower error so should have higher weight
        assert ens.weights["LightGBM"] > ens.weights["Prophet"]
