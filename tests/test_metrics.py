"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from diffsurrogate.evaluation.metrics import (
    compute_metrics,
    constraint_violation_rate,
    coverage_95,
    mae,
    nll,
    rmse,
)


class TestRMSE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_known_value(self):
        """sqrt(mean([0.01, 0.01, 0.04, 0.01])) = sqrt(0.0175)."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.9])
        expected = math.sqrt((0.01 + 0.01 + 0.04 + 0.01) / 4.0)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-12

    def test_nan_safe(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        with pytest.warns(RuntimeWarning):
            r = rmse(y_true, y_pred)
        assert r == 0.0  # Only the 2 finite rows, both exact.

    def test_all_nan_returns_nan(self):
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([1.0, 2.0])
        with pytest.warns(RuntimeWarning):
            r = rmse(y_true, y_pred)
        assert math.isnan(r)


class TestMAE:
    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        # mean(|0.5|, |0.5|, |0.5|) = 0.5
        assert abs(mae(y_true, y_pred) - 0.5) < 1e-12


class TestNLL:
    def test_returns_nan_without_std(self):
        y = np.array([1.0, 2.0])
        assert math.isnan(nll(y, y, None))

    def test_known_value_perfect_prediction(self):
        """Perfect prediction with unit std: NLL = 0.5 * log(2π) per sample."""
        y = np.array([0.0, 0.0, 0.0])
        y_std = np.array([1.0, 1.0, 1.0])
        expected = 0.5 * math.log(2.0 * math.pi)
        assert abs(nll(y, y, y_std) - expected) < 1e-10

    def test_floors_nonpositive_std(self):
        """Zero std shouldn't crash; it floors to 1e-12."""
        y_true = np.array([1.0])
        y_pred = np.array([1.0])
        y_std = np.array([0.0])
        val = nll(y_true, y_pred, y_std)
        assert np.isfinite(val)


class TestCoverage95:
    def test_all_inside(self):
        """y_true = y_pred → all inside regardless of std."""
        y = np.array([0.0, 0.0, 0.0])
        s = np.array([1.0, 1.0, 1.0])
        assert coverage_95(y, y, s) == 1.0

    def test_half_inside(self):
        """Alternate error 1.0 / 3.0 with std=1 → first two inside, third outside."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.5, 3.0])
        y_std = np.array([1.0, 1.0, 1.0])
        # Errors: 1.0 <= 2, 1.5 <= 2, 3.0 > 2 → 2/3 inside
        assert abs(coverage_95(y_true, y_pred, y_std) - (2.0 / 3.0)) < 1e-12

    def test_returns_nan_without_std(self):
        y = np.array([1.0])
        assert math.isnan(coverage_95(y, y, None))


class TestConstraintViolationRate:
    def test_none_input(self):
        assert math.isnan(constraint_violation_rate(None))

    def test_all_in_bounds(self):
        arr = np.array([0.0, 0.5, 1.0])
        assert constraint_violation_rate(arr) == 0.0

    def test_all_out_of_bounds(self):
        arr = np.array([-0.1, 1.1, 2.0])
        assert constraint_violation_rate(arr) == 1.0

    def test_half_out(self):
        arr = np.array([0.5, 1.5, 0.3, -0.2])
        # 2 out of 4 violate: 1.5, -0.2
        assert abs(constraint_violation_rate(arr) - 0.5) < 1e-12

    def test_uses_n_pred_preferred(self):
        """If n_pred is supplied, use it instead of y_pred_physical."""
        # physical: all good; n: all bad
        phys = np.array([0.1, 0.2, 0.3])
        n = np.array([-1.0, 2.0, 3.0])
        assert constraint_violation_rate(phys, n_pred=n) == 1.0


class TestComputeMetrics:
    def test_dispatch_subset(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        y_std = np.array([0.1, 0.1, 0.1])
        out = compute_metrics(y_true, y_pred, y_std, enabled=["rmse", "mae"])
        assert set(out) == {"rmse", "mae"}
        assert abs(out["rmse"] - math.sqrt((0.01 + 0.01 + 0.04) / 3)) < 1e-12

    def test_dispatch_all_metrics(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        y_std = np.array([0.5, 0.5])
        out = compute_metrics(
            y_true, y_pred, y_std,
            y_pred_physical=np.array([0.3, 0.4]),
            enabled=["rmse", "nll", "coverage_95", "constraint_violation_rate"],
        )
        assert out["rmse"] == 0.0
        assert out["coverage_95"] == 1.0
        assert out["constraint_violation_rate"] == 0.0
        assert np.isfinite(out["nll"])

    def test_unknown_metric_warns_and_returns_nan(self):
        out = compute_metrics(
            np.array([1.0]), np.array([1.0]),
            enabled=["rmse", "bogus_metric"],
        )
        assert out["rmse"] == 0.0
        assert math.isnan(out["bogus_metric"])

    def test_nll_nan_when_no_std_in_dispatcher(self):
        out = compute_metrics(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), y_std=None,
            enabled=["nll", "coverage_95"],
        )
        assert math.isnan(out["nll"])
        assert math.isnan(out["coverage_95"])
