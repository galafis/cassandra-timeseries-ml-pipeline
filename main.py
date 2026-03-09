"""
Cassandra Time Series ML Pipeline -- Demo Runner
=================================================

Generates synthetic retail sales data with realistic patterns (trend,
weekly seasonality, annual seasonality, promotional spikes, and noise),
then runs the full pipeline end to end:

1. Store data in the local Cassandra simulation layer.
2. Generate lag, rolling, calendar, and Fourier features.
3. Train Prophet (statsmodels fallback), LightGBM, and Ensemble models.
4. Evaluate on a held-out test window and cross-validate.
5. Demonstrate hierarchical forecast reconciliation.
6. Print a comparison report and a sample forecast table.

No external infrastructure (Cassandra, Spark) is required -- everything
runs locally with pandas, numpy, scikit-learn, and lightgbm.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.storage.cassandra_client import CassandraClient
from src.features.feature_generator import TimeSeriesFeatureGenerator
from src.models.prophet_model import ProphetForecaster
from src.models.lightgbm_model import LightGBMForecaster
from src.models.ensemble_model import EnsembleForecaster
from src.training.trainer import PipelineTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.reconciliation.reconciler import HierarchicalReconciler
from src.utils.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("main")

# ======================================================================
# 1.  Synthetic data generation
# ======================================================================

def generate_synthetic_sales(
    n_days: int = 730,
    start_date: str = "2022-01-01",
    base_sales: float = 1_000.0,
    trend_slope: float = 0.5,
    noise_std: float = 50.0,
    series_id: str = "store_1",
) -> pd.DataFrame:
    """
    Create a realistic daily retail sales time series.

    Components
    ----------
    - Linear upward trend
    - Weekly seasonality (higher on weekends)
    - Annual seasonality (holiday peaks in Dec, summer dip)
    - Random promotional spikes (~5 % of days)
    - Gaussian noise
    """
    rng = np.random.RandomState(42)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    t = np.arange(n_days, dtype=np.float64)

    trend = base_sales + trend_slope * t

    # Weekly pattern -- weekends are ~20 % higher
    day_of_week = dates.dayofweek
    weekly = np.where(day_of_week >= 5, 0.20, 0.0) * base_sales

    # Annual seasonality
    day_of_year = dates.dayofyear
    annual = (
        150 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi / 2)
        + 100 * np.cos(4 * np.pi * day_of_year / 365.25)
    )

    # Promotional spikes on ~5 % of days
    promo_mask = rng.rand(n_days) < 0.05
    promo_effect = promo_mask * rng.uniform(200, 600, size=n_days)

    noise = rng.normal(0, noise_std, size=n_days)

    values = trend + weekly + annual + promo_effect + noise
    values = np.maximum(values, 0)  # sales cannot be negative

    return pd.DataFrame({
        "timestamp": dates,
        "value": np.round(values, 2),
        "series_id": series_id,
    })


def generate_hierarchical_data(n_days: int = 365) -> pd.DataFrame:
    """Generate multi-store data for reconciliation demo."""
    frames = []
    configs = [
        ("store_1", 1_000, 0.5, 50),
        ("store_2", 800, 0.3, 40),
        ("store_3", 600, 0.2, 30),
    ]
    for sid, base, slope, noise in configs:
        df = generate_synthetic_sales(
            n_days=n_days,
            base_sales=base,
            trend_slope=slope,
            noise_std=noise,
            series_id=sid,
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


# ======================================================================
# 2.  Main pipeline
# ======================================================================

def run_demo() -> None:
    """Execute the full demonstration pipeline."""
    print("\n" + "=" * 70)
    print("   Cassandra Time Series ML Pipeline -- Demo")
    print("=" * 70 + "\n")

    # ---- 2a. Generate synthetic data --------------------------------
    print("[1/7] Generating synthetic retail sales data ...")
    raw_df = generate_synthetic_sales(n_days=730, series_id="store_1")
    print(f"      Generated {len(raw_df)} daily observations")
    print(f"      Date range: {raw_df['timestamp'].min().date()} to "
          f"{raw_df['timestamp'].max().date()}")
    print(f"      Value range: {raw_df['value'].min():.0f} - "
          f"{raw_df['value'].max():.0f}\n")

    # ---- 2b. Store in local simulation ------------------------------
    print("[2/7] Storing data in Cassandra (simulation mode) ...")
    client = CassandraClient()
    client.connect()
    client.create_keyspace()
    client.create_table()
    inserted = client.insert_dataframe(raw_df)
    retrieved = client.query_timeseries("store_1")
    print(f"      Inserted {inserted} rows, retrieved {len(retrieved)} rows\n")

    # ---- 2c. Feature engineering ------------------------------------
    print("[3/7] Generating features ...")
    feat_gen = TimeSeriesFeatureGenerator(
        lag_orders=[1, 2, 3, 7, 14, 28],
        rolling_windows=[7, 14, 30],
        fourier_order=4,
    )
    featured_df = feat_gen.generate_all_features(raw_df, drop_na=True)
    feature_cols = feat_gen.get_feature_columns(featured_df)
    print(f"      {len(feature_cols)} features, {len(featured_df)} rows "
          f"after dropping NaN\n")

    # ---- 2d. Train models -------------------------------------------
    print("[4/7] Training models ...")
    prophet = ProphetForecaster()
    lgbm = LightGBMForecaster(params={
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    })
    ensemble = EnsembleForecaster(
        models=[prophet, lgbm],
        weights={"Prophet": 0.4, "LightGBM": 0.6},
        method="weighted_average",
    )

    trainer = PipelineTrainer(
        models=[prophet, lgbm, ensemble],
        feature_generator=feat_gen,
        n_folds=3,
        train_ratio=0.8,
    )

    results = trainer.run_training_pipeline(raw_df, run_cv=True)

    # ---- 2e. Evaluation report --------------------------------------
    print("\n[5/7] Evaluating models ...")
    evaluator = ModelEvaluator()
    report = evaluator.generate_report(
        results["final_metrics"],
        cv_results=results.get("cv_results"),
        title="Retail Sales Forecast -- Model Comparison",
    )
    print(report)

    # ---- 2f. Sample forecast ----------------------------------------
    print("[6/7] Generating 14-day forecast ...")
    horizon = 14
    best_name = min(
        results["final_metrics"],
        key=lambda m: results["final_metrics"][m].get("rmse", float("inf")),
    )
    best_model = results["models"][best_name]

    # Build future features for LightGBM-based models
    last_rows = results["train_df"].tail(horizon).copy()
    future_features = feat_gen.generate_all_features(
        pd.concat([raw_df.tail(90), last_rows]).drop_duplicates(),
        drop_na=True,
    ).tail(horizon)

    forecast_df = best_model.predict(horizon=horizon, future_df=future_features)
    print(f"\n      Best model: {best_name}")
    print(f"      Forecast horizon: {horizon} days\n")

    forecast_display = forecast_df.copy()
    forecast_display["timestamp"] = pd.to_datetime(forecast_display["timestamp"]).dt.date
    forecast_display["forecast"] = forecast_display["forecast"].round(2)
    print(forecast_display.to_string(index=False))

    # ---- 2g. Hierarchical reconciliation demo -----------------------
    print(f"\n[7/7] Hierarchical reconciliation demo ...")
    hierarchy = {
        "total": ["region_A", "region_B"],
        "region_A": ["store_1", "store_2"],
        "region_B": ["store_3"],
    }
    reconciler = HierarchicalReconciler(
        hierarchy=hierarchy, method="bottom_up", non_negative=True
    )

    # Fake per-store forecasts for demonstration
    rng = np.random.RandomState(99)
    base_forecasts = {
        "total": rng.uniform(2000, 3000, size=horizon),
        "region_A": rng.uniform(1200, 1800, size=horizon),
        "region_B": rng.uniform(500, 900, size=horizon),
        "store_1": rng.uniform(600, 1000, size=horizon),
        "store_2": rng.uniform(400, 800, size=horizon),
        "store_3": rng.uniform(500, 900, size=horizon),
    }

    reconciled = reconciler.reconcile(base_forecasts, method="bottom_up")
    print(f"\n      Reconciled {len(reconciled)} series (bottom-up)")
    print(f"      Total forecast (day 1): base={base_forecasts['total'][0]:.0f}, "
          f"reconciled={reconciled['total'][0]:.0f}")

    # Also show MinT reconciliation
    reconciled_mint = reconciler.reconcile(
        base_forecasts, method="mint"
    )
    print(f"      Total forecast (day 1, MinT): reconciled={reconciled_mint['total'][0]:.0f}")

    print("\n" + "=" * 70)
    print("   Pipeline complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
