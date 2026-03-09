"""
Model evaluation and comparison module.

Provides a consistent interface for computing regression metrics,
ranking models, and producing human-readable comparison reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates and compares time series forecasting models.

    Stores evaluation history so multiple runs can be compared in a
    single report.
    """

    def __init__(self) -> None:
        self._results: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_metrics(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute a comprehensive set of forecasting metrics.

        Parameters
        ----------
        actual : array-like
            Ground-truth values.
        predicted : array-like
            Model predictions.

        Returns
        -------
        dict
            Keys: ``rmse``, ``mae``, ``mape``, ``smape``, ``r2``,
            ``directional_accuracy``.
        """
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)

        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        r2 = float(r2_score(actual, predicted))

        nonzero = actual != 0
        if nonzero.any():
            mape = float(
                np.mean(
                    np.abs(
                        (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
                    )
                )
                * 100
            )
        else:
            mape = float("inf")

        denom = np.abs(actual) + np.abs(predicted)
        safe = np.where(denom == 0, 1.0, denom)
        smape = float(np.mean(2.0 * np.abs(actual - predicted) / safe) * 100)

        if len(actual) > 1:
            dir_actual = np.diff(actual) > 0
            dir_pred = np.diff(predicted) > 0
            directional_accuracy = float(np.mean(dir_actual == dir_pred) * 100)
        else:
            directional_accuracy = 0.0

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 4),
            "smape": round(smape, 4),
            "r2": round(r2, 4),
            "directional_accuracy": round(directional_accuracy, 4),
        }

    # ------------------------------------------------------------------
    # Multi-model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
        sort_by: str = "rmse",
        ascending: bool = True,
    ) -> pd.DataFrame:
        """
        Build a comparison table from per-model metric dictionaries.

        Parameters
        ----------
        results : dict[str, dict[str, float]]
            ``{model_name: metrics_dict}``.
        sort_by : str
            Metric used for ranking.
        ascending : bool
            Sort direction (True = lower is better, e.g. RMSE).

        Returns
        -------
        DataFrame
            One row per model, sorted by the chosen metric.
        """
        self._results.update(results)

        rows = []
        for model_name, metrics in results.items():
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"
        return df

    def generate_report(
        self,
        results: Dict[str, Dict[str, float]],
        cv_results: Optional[Dict[str, List[Dict[str, float]]]] = None,
        title: str = "Model Evaluation Report",
    ) -> str:
        """
        Produce a plain-text evaluation report.

        Parameters
        ----------
        results : dict
            Final test-set metrics per model.
        cv_results : dict | None
            Cross-validation fold results for mean/std reporting.
        title : str
            Report header text.

        Returns
        -------
        str
            Multi-line report string.
        """
        lines: List[str] = []
        sep = "=" * 70

        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)
        lines.append("")

        # Final metrics table
        comparison = self.compare_models(results)
        lines.append("Final Test Metrics (sorted by RMSE):")
        lines.append("-" * 70)
        lines.append(comparison.to_string())
        lines.append("")

        # Cross-validation summary
        if cv_results:
            lines.append("Cross-Validation Summary (mean +/- std):")
            lines.append("-" * 70)

            for model_name, folds in cv_results.items():
                if not folds:
                    continue
                fold_df = pd.DataFrame(folds)
                numeric_cols = fold_df.select_dtypes(include=[np.number]).columns
                means = fold_df[numeric_cols].mean()
                stds = fold_df[numeric_cols].std()

                lines.append(f"\n  {model_name}:")
                for col in numeric_cols:
                    if col == "fold":
                        continue
                    lines.append(
                        f"    {col:>22s}: {means[col]:>10.4f} +/- {stds[col]:.4f}"
                    )

        lines.append("")

        # Best model
        if results:
            best_model = min(results, key=lambda m: results[m].get("rmse", float("inf")))
            best_rmse = results[best_model]["rmse"]
            lines.append(f"Best model: {best_model} (RMSE = {best_rmse:.4f})")

        lines.append(sep)

        report = "\n".join(lines)
        logger.info("Evaluation report generated (%d characters)", len(report))
        return report

    def aggregate_cv_metrics(
        self,
        cv_results: Dict[str, List[Dict[str, float]]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute mean metrics across CV folds for each model.

        Returns
        -------
        dict
            ``{model_name: {metric: mean_value, ...}}``
        """
        aggregated: Dict[str, Dict[str, float]] = {}

        for model_name, folds in cv_results.items():
            if not folds:
                continue
            fold_df = pd.DataFrame(folds)
            numeric = fold_df.select_dtypes(include=[np.number])
            means = numeric.mean()
            aggregated[model_name] = {
                k: round(v, 4)
                for k, v in means.items()
                if k != "fold"
            }

        return aggregated
