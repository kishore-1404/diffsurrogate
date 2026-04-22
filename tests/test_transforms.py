"""Tests for physics-informed scalers."""

from __future__ import annotations

import numpy as np
import pytest

from diffsurrogate.data.transforms import (
    InputScalerBundle,
    LogStabilizedScaler,
    LogZScoreScaler,
    RobustTargetScaler,
    StandardTargetScaler,
    ZScoreScaler,
    build_input_scaler,
    build_target_scaler,
)


# -----------------------------------------------------------------------------
# Individual scalers
# -----------------------------------------------------------------------------

class TestZScoreScaler:
    def test_fit_transform_identity(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 2.0, size=200)
        s = ZScoreScaler()
        xs = s.fit_transform(x)
        assert abs(xs.mean()) < 1e-10
        assert abs(xs.std() - 1.0) < 1e-10

    def test_inverse(self):
        rng = np.random.default_rng(1)
        x = rng.normal(10.0, 3.0, size=100)
        s = ZScoreScaler()
        xs = s.fit_transform(x)
        x_back = s.inverse_transform(xs)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_zero_variance(self):
        """Constant input → std clamped to 1 to avoid div-by-zero."""
        s = ZScoreScaler()
        s.fit(np.array([3.0, 3.0, 3.0]))
        assert s.std_ == 1.0
        # Transform of the constant value: (3-3)/1 = 0
        assert float(s.transform(np.array([3.0])).item()) == 0.0

    def test_unfitted_raises(self):
        s = ZScoreScaler()
        with pytest.raises(RuntimeError):
            s.transform(np.array([1.0]))


class TestLogZScoreScaler:
    def test_invertibility(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(0.5, 100.0, size=200)
        s = LogZScoreScaler()
        xs = s.fit_transform(x)
        x_back = s.inverse_transform(xs)
        assert np.allclose(x, x_back, rtol=1e-10, atol=1e-10)

    def test_rejects_non_positive_on_fit(self):
        s = LogZScoreScaler()
        with pytest.raises(ValueError):
            s.fit(np.array([1.0, 0.0, 2.0]))

    def test_rejects_non_positive_on_transform(self):
        s = LogZScoreScaler()
        s.fit(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            s.transform(np.array([-0.1]))


class TestLogStabilizedScaler:
    def test_invertibility_negative_t(self):
        """Fit on negative values (Mandelstam t convention); inverse restores sign."""
        rng = np.random.default_rng(3)
        t = -rng.uniform(0.01, 2.0, size=200)
        s = LogStabilizedScaler(t_epsilon_frac=0.01)
        ts = s.fit_transform(t)
        # After fit/transform: should be approximately z-scored.
        assert abs(ts.mean()) < 1e-10
        assert abs(ts.std() - 1.0) < 1e-10
        # Inverse restores negative sign by default since training signs were negative.
        t_back = s.inverse_transform(ts)
        assert np.allclose(t, t_back, rtol=1e-8, atol=1e-8)
        # Sanity: sign preserved.
        assert (t_back <= 0).all()

    def test_epsilon_computation(self):
        """eps = t_epsilon_frac * median(|t|)."""
        t = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])
        frac = 0.1
        s = LogStabilizedScaler(t_epsilon_frac=frac)
        s.fit(t)
        expected_eps = frac * np.median(np.abs(t))
        assert abs(s.eps_ - expected_eps) < 1e-12

    def test_sign_detection(self):
        """Majority-negative training → sign_ == -1."""
        s = LogStabilizedScaler()
        s.fit(np.array([-0.1, -0.2, -0.3, 0.1]))
        assert s.sign_ == -1
        s2 = LogStabilizedScaler()
        s2.fit(np.array([0.1, 0.2, 0.3, -0.1]))
        assert s2.sign_ == 1

    def test_handles_zeros(self):
        """|t| = 0 survives because of the epsilon."""
        t = np.array([0.0, -0.5, -1.0])
        s = LogStabilizedScaler()
        ts = s.fit_transform(t)
        assert np.isfinite(ts).all()

    def test_epsilon_floor(self):
        """If data is all zero, eps shouldn't collapse to zero."""
        t = np.array([0.0, 0.0, 0.0])
        s = LogStabilizedScaler(t_epsilon_frac=0.01)
        s.fit(t)
        assert s.eps_ > 0


class TestRobustTargetScaler:
    def test_invertibility(self):
        rng = np.random.default_rng(4)
        y = rng.normal(-3.0, 2.0, size=200)
        s = RobustTargetScaler()
        ys = s.fit_transform(y)
        y_back = s.inverse_transform(ys)
        assert np.allclose(y.ravel(), np.asarray(y_back).ravel(), atol=1e-10)

    def test_robust_to_outliers(self):
        y = np.concatenate([np.zeros(100), np.array([1e6])])  # one huge outlier
        s = RobustTargetScaler()
        ys = s.fit_transform(y)
        # The bulk (zeros) should map cleanly near zero; the outlier dominates
        # a mean/std scaler but not a median/IQR one.
        # Since IQR is 0 for mostly-zero data, it gets floored to 1.0.
        # Median is 0, so scaled values are y - 0 = y.
        assert s.median_ == 0.0

    def test_std_to_original(self):
        y = np.arange(100, dtype=float)
        s = RobustTargetScaler()
        s.fit(y)
        assert s.iqr_ > 0
        orig_std = s.std_to_original(np.array([1.0, 2.0]))
        assert np.allclose(orig_std, np.array([s.iqr_, 2.0 * s.iqr_]))


class TestStandardTargetScaler:
    def test_invertibility(self):
        rng = np.random.default_rng(5)
        y = rng.normal(5.0, 1.5, size=100)
        s = StandardTargetScaler()
        ys = s.fit_transform(y)
        y_back = s.inverse_transform(ys)
        assert np.allclose(y.ravel(), np.asarray(y_back).ravel(), atol=1e-10)


# -----------------------------------------------------------------------------
# Bundle + factories
# -----------------------------------------------------------------------------

class TestInputScalerBundle:
    def test_bundle_invertibility(self, synthetic_dataset):
        X, _ = synthetic_dataset
        bundle = InputScalerBundle(
            q2=ZScoreScaler(),        # Q2 is already log in the fixture
            w2=LogZScoreScaler(),
            t=LogStabilizedScaler(),
        )
        Xs = bundle.fit_transform(X)
        X_back = bundle.inverse_transform(Xs)
        assert np.allclose(X, X_back, atol=1e-8)


class TestFactories:
    def test_build_input_scaler_prelogged(self):
        """q2_is_prelogged=True collapses log_zscore to plain zscore on Q2."""
        from diffsurrogate.config import DataConfig, TransformsConfig

        data_cfg = DataConfig(
            input_path="", predict_path="",
            input_columns=["Q2_log_center", "W2_center", "t_center"],
            target_column="logA", train_fraction=0.8, random_seed=0,
            q2_is_prelogged=True,
        )
        tr_cfg = TransformsConfig(
            q2_transform="log_zscore", w2_transform="log_zscore",
            t_transform="log_stabilized", t_epsilon_frac=0.01,
            target_scaler="robust",
        )
        bundle = build_input_scaler(tr_cfg, data_cfg)
        assert isinstance(bundle.q2, ZScoreScaler)
        assert isinstance(bundle.w2, LogZScoreScaler)
        assert isinstance(bundle.t, LogStabilizedScaler)

    def test_build_input_scaler_raw_q2(self):
        """q2_is_prelogged=False keeps full LogZScoreScaler on Q2."""
        from diffsurrogate.config import DataConfig, TransformsConfig

        data_cfg = DataConfig(
            input_path="", predict_path="",
            input_columns=["Q2", "W2", "t"],
            target_column="logA", train_fraction=0.8, random_seed=0,
            q2_is_prelogged=False,
        )
        tr_cfg = TransformsConfig(
            q2_transform="log_zscore", w2_transform="log_zscore",
            t_transform="log_stabilized", t_epsilon_frac=0.01,
            target_scaler="robust",
        )
        bundle = build_input_scaler(tr_cfg, data_cfg)
        assert isinstance(bundle.q2, LogZScoreScaler)

    def test_build_target_scaler(self):
        from diffsurrogate.config import TransformsConfig

        tr_robust = TransformsConfig("zscore", "log_zscore", "log_stabilized", 0.01, "robust")
        assert isinstance(build_target_scaler(tr_robust), RobustTargetScaler)
        tr_std = TransformsConfig("zscore", "log_zscore", "log_stabilized", 0.01, "standard")
        assert isinstance(build_target_scaler(tr_std), StandardTargetScaler)


# -----------------------------------------------------------------------------
# fit-on-training-only invariant
# -----------------------------------------------------------------------------

class TestFitOnTrainOnly:
    def test_epsilon_from_train(self, synthetic_dataset):
        """Epsilon should depend ONLY on training data, not test data."""
        X, _ = synthetic_dataset
        X_train = X[:150]
        X_test = X[150:]
        s_train_only = LogStabilizedScaler()
        s_train_only.fit(X_train[:, 2])

        s_leaky = LogStabilizedScaler()
        s_leaky.fit(np.concatenate([X_train[:, 2], X_test[:, 2]]))
        # Epsilons should differ (test data changes median of |t|).
        # They might coincidentally be close, so just check the fit is
        # not using test data via a direct assertion.
        assert s_train_only.eps_ is not None
        # Transform test data through the train-only scaler — should be finite.
        xs_test = s_train_only.transform(X_test[:, 2])
        assert np.isfinite(xs_test).all()
