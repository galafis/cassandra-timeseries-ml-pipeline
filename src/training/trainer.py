"""
Pipeline trainer with time-series-aware cross-validation.

Orchestrates data preparation, model training, and evaluation using
expanding-window cross-validation to respect the temporal ordering
of observations.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.feature_generator import TimeSeriesFeatureGenerator
from src.models.base_model import BaseTimeSeriesModel
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineTrainer:
    """
    End-to-end training orchestrator.

    Handles feature generation, train/test splitting, cross-validation
    with expanding windows, and metric collection for every registered
    model.

    Parameters
    ----------
    models : list[BaseTimeSeriesModel]
        Models to train and evaluate.
    feature_generator : TimeSeriesFeatureGenerator
        Feature engineering instance.
    n_folds : int
        Number of cross-validation folds.
    train_ratio : float
        Fraction of data used for the final training split.
    date_column : str
        Name of the datetime column.
    target_column : str
        Name of the target value column.
    """

    def __init__(
        self,
        models: List[BaseTimeSeriesModel],
        feature_generator: Optional[TimeSeriesFeatureGenerator] = None,
        n_folds: int = 5,
        train_ratio: float = 0.8,
        date_column: str = "timestamp",
        target_column: str = "value",
    ) -> None:
        self.models = models
        self.feature_generator = feature_generator or TimeSeriesFeatureGenerator()
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.date_column = date_column
        self.target_column = target_column
        self.evaluator = ModelEvaluator()

        self.cv_results: Dict[str, List[Dict[str, float]]] = {}
        self.final_metrics: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        generate_features: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sort, generate features, and split data into train / test.

        Returns
        -------
        train_df, test_df : tuple[DataFrame, DataFrame]
        """
        data = df.copy()
        data[self.date_column] = pd.to_datetime(data[self.date_column])
        data = data.sort_values(self.date_column).reset_index(drop=True)

        if generate_features:
            data = self.feature_generator.generate_all_features(
                data, drop_na=True
            )

        split_idx = int(len(data) * self.train_ratio)
        train_df = data.iloc[:split_idx].reset_index(drop=True)
        test_df = data.iloc[split_idx:].reset_index(drop=True)

        logger.info(
            "Data split: %d train rows, %d test rows",
            len(train_df),
            len(test_df),
        )
        return train_df, test_df

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_models(
        self,
        train_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, BaseTimeSeriesModel]:
        """
        Fit every registered model on the training data.

        Returns a mapping of model name to fitted model instance.
        """
        if feature_columns is None:
            feature_columns = self.feature_generator.get_feature_columns(train_df)

        trained: Dict[str, BaseTimeSeriesModel] = {}

        for model in self.models:
            logger.info("Training model '%s'...", model.name)
            model.fit(
                train_df,
                date_column=self.date_column,
                target_column=self.target_column,
                feature_columns=feature_columns,
            )
            trained[model.name] = model

        logger.info("All %d models trained", len(trained))
        return trained

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Run expanding-window cross-validation.

        The dataset is split into ``n_folds + 1`` sequential blocks.
        In fold *i*, blocks ``0..i`` form the training set and block
        ``i + 1`` is used for validation.

        Returns
        -------
        dict
            ``{model_name: [fold_metrics, ...]}``
        """
        data = df.copy()
        data[self.date_column] = pd.to_datetime(data[self.date_column])
        data = data.sort_values(self.date_column).reset_index(drop=True)

        n = len(data)
        fold_size = n // (self.n_folds + 1)

        if fold_size < 10:
            logger.warning(
                "Very small fold size (%d rows). Reducing folds.", fold_size
            )
            self.n_folds = max(1, n // 10 - 1)
            fold_size = n // (self.n_folds + 1)

        if feature_columns is None:
            feature_columns = self.feature_generator.get_feature_columns(data)

        self.cv_results = {m.name: [] for m in self.models}

        for fold_idx in range(self.n_folds):
            train_end = fold_size * (fold_idx + 1)
            val_start = train_end
            val_end = min(val_start + fold_size, n)

            if val_end <= val_start:
                break

            fold_train = data.iloc[:train_end].reset_index(drop=True)
            fold_val = data.iloc[val_start:val_end].reset_index(drop=True)

            logger.info(
                "CV fold %d/%d: train=%d, val=%d",
                fold_idx + 1,
                self.n_folds,
                len(fold_train),
                len(fold_val),
            )

            for model in self.models:
                fold_model = self._clone_model(model)
                try:
                    fold_model.fit(
                        fold_train,
                        date_column=self.date_column,
                        target_column=self.target_column,
                        feature_columns=feature_columns,
                    )

                    preds = fold_model.predict(
                        horizon=len(fold_val), future_df=fold_val
                    )
                    forecast_vals = preds["forecast"].values[: len(fold_val)]
                    actual_vals = fold_val[self.target_column].values[: len(forecast_vals)]

                    metrics = fold_model.evaluate(actual_vals, forecast_vals)
                    metrics["fold"] = fold_idx + 1
                    self.cv_results[model.name].append(metrics)

                except Exception as exc:
                    logger.warning(
                        "Fold %d failed for '%s': %s",
                        fold_idx + 1,
                        model.name,
                        exc,
                    )

        logger.info("Cross-validation complete (%d folds)", self.n_folds)
        return self.cv_results

    @staticmethod
    def _clone_model(model: BaseTimeSeriesModel) -> BaseTimeSeriesModel:
        """Create a fresh copy of a model for fold-level training."""
        cloned = copy.deepcopy(model)
        cloned._is_fitted = False
        return cloned

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_training_pipeline(
        self,
        raw_df: pd.DataFrame,
        run_cv: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.

        1. Generate features and split data.
        2. Optionally run expanding-window cross-validation.
        3. Train final models on the full training set.
        4. Evaluate on the held-out test set.

        Returns
        -------
        dict
            Contains ``"cv_results"``, ``"final_metrics"``,
            ``"train_df"``, ``"test_df"``, and ``"models"``.
        """
        train_df, test_df = self.prepare_data(raw_df, generate_features=True)

        feature_cols = self.feature_generator.get_feature_columns(train_df)

        cv_results: Dict[str, List[Dict[str, float]]] = {}
        if run_cv and len(train_df) > 50:
            cv_results = self.cross_validate(train_df, feature_columns=feature_cols)

        trained_models = self.train_models(train_df, feature_columns=feature_cols)

        # Evaluate on test set
        self.final_metrics = {}
        for name, model in trained_models.items():
            try:
                preds = model.predict(horizon=len(test_df), future_df=test_df)
                forecast_vals = preds["forecast"].values[: len(test_df)]
                actual_vals = test_df[self.target_column].values[: len(forecast_vals)]
                self.final_metrics[name] = model.evaluate(actual_vals, forecast_vals)
            except Exception as exc:
                logger.warning("Test evaluation failed for '%s': %s", name, exc)

        return {
            "cv_results": cv_results,
            "final_metrics": self.final_metrics,
            "train_df": train_df,
            "test_df": test_df,
            "models": trained_models,
        }
