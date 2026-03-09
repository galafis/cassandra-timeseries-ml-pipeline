"""
Unit tests for the ModelEvaluator.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import ModelEvaluator


class TestCalculateMetrics:
    """Tests for the static calculate_metrics helper."""

    def test_perfect_prediction(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = actual.copy()
        metrics = ModelEvaluator.calculate_metrics(actual, predicted)

        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["mape"] == pytest.approx(0.0, abs=1e-2)
        assert metrics["smape"] == pytest.approx(0.0, abs=1e-2)

    def test_constant_offset(self) -> None:
        actual = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        predicted = actual + 5.0
        metrics = ModelEvaluator.calculate_metrics(actual, predicted)

        assert metrics["rmse"] == pytest.approx(5.0, abs=1e-4)
        assert metrics["mae"] == pytest.approx(5.0, abs=1e-4)
        assert metrics["mape"] > 0
        assert metrics["r2"] > 0  # Still highly correlated

    def test_inverse_prediction_has_negative_r2(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        metrics = ModelEvaluator.calculate_metrics(actual, predicted)

        assert metrics["r2"] < 0

    def test_directional_accuracy_all_correct(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        metrics = ModelEvaluator.calculate_metrics(actual, predicted)

        assert metrics["directional_accuracy"] == pytest.approx(100.0)

    def test_directional_accuracy_all_wrong(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        metrics = ModelEvaluator.calculate_metrics(actual, predicted)

        assert metrics["directional_accuracy"] == pytest.approx(0.0)

    def test_handles_zeros_in_actual(self) -> None:
        actual = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        predicted = np.array([0.1, 1.1, 2.1, 0.1, 3.1])
        metrics = ModelEvaluator.calculate_metrics(actual, predicted)

        # MAPE is computed only on non-zero actuals
        assert not math.isinf(metrics["mape"])
        assert metrics["mape"] > 0

    def test_mismatched_lengths_truncated(self) -> None:
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = ModelEvaluator.calculate_metrics(actual, predicted)

        # Should use the first 3 values only (perfect match)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)


class TestCompareModels:
    """Tests for the model comparison table."""

    def test_comparison_sorted_by_rmse(self) -> None:
        evaluator = ModelEvaluator()
        results = {
            "ModelA": {"rmse": 10.0, "mae": 8.0, "r2": 0.8},
            "ModelB": {"rmse": 5.0, "mae": 4.0, "r2": 0.9},
            "ModelC": {"rmse": 15.0, "mae": 12.0, "r2": 0.6},
        }
        table = evaluator.compare_models(results, sort_by="rmse")

        assert table.iloc[0]["model"] == "ModelB"
        assert table.iloc[-1]["model"] == "ModelC"

    def test_comparison_has_all_models(self) -> None:
        evaluator = ModelEvaluator()
        results = {
            "A": {"rmse": 1.0},
            "B": {"rmse": 2.0},
        }
        table = evaluator.compare_models(results)
        assert set(table["model"]) == {"A", "B"}


class TestGenerateReport:
    """Tests for the text report generator."""

    def test_report_contains_best_model(self) -> None:
        evaluator = ModelEvaluator()
        results = {
            "Good": {"rmse": 2.0, "mae": 1.5, "r2": 0.95},
            "Bad": {"rmse": 20.0, "mae": 15.0, "r2": 0.2},
        }
        report = evaluator.generate_report(results)

        assert "Good" in report
        assert "Best model" in report

    def test_report_includes_cv_section(self) -> None:
        evaluator = ModelEvaluator()
        results = {"M": {"rmse": 5.0}}
        cv = {"M": [{"rmse": 5.1, "fold": 1}, {"rmse": 4.9, "fold": 2}]}
        report = evaluator.generate_report(results, cv_results=cv)

        assert "Cross-Validation" in report

    def test_report_is_nonempty_string(self) -> None:
        evaluator = ModelEvaluator()
        results = {"X": {"rmse": 1.0}}
        report = evaluator.generate_report(results)
        assert isinstance(report, str)
        assert len(report) > 50
