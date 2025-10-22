from __future__ import annotations

import numpy as np
import pandas as pd

from amie.models.kalman import KalmanFilter


def _make_features(
    returns: np.ndarray,
    *,
    instrument: str = "BTC-USD",
    start_ts: str = "2024-01-01",
) -> pd.DataFrame:
    ts = pd.date_range(start_ts, periods=len(returns), freq="s")
    zeros = np.zeros_like(returns)
    return pd.DataFrame(
        {
            "ts": ts,
            "instrument": instrument,
            "returns": returns,
            "ewma_volatility": zeros,
            "z_score": zeros,
            "spread": zeros,
            "imbalance": zeros,
        }
    )


def test_kalman_converges_to_true_trend() -> None:
    rng = np.random.default_rng(123)
    true_trend = 0.001
    returns = true_trend + rng.normal(0.0, 0.0002, size=120)
    features = _make_features(returns)

    model = KalmanFilter(instrument="BTC-USD", process_noise=1e-4, observation_noise=5e-3, warmup_period=10)
    model.fit(features.iloc[:20])
    signals = model.predict(features.iloc[20:])

    scores = np.array([signal.score for signal in signals])
    assert np.isfinite(scores).all()
    assert abs(scores[-1] - true_trend) < 3e-4


def test_uncertainty_decreases_over_time() -> None:
    rng = np.random.default_rng(42)
    returns = 0.0005 + rng.normal(0, 0.0002, size=100)
    features = _make_features(returns)

    model = KalmanFilter(instrument="BTC-USD", process_noise=1e-4, observation_noise=1e-2, warmup_period=5)
    model.fit(features.iloc[:10])
    signals = model.predict(features.iloc[10:])

    uncertainties = np.array([signal.uncertainty for signal in signals])
    assert uncertainties[0] > uncertainties[-1]
    assert np.all(uncertainties >= 0)


def test_uncertainty_increases_on_volatility_shift() -> None:
    rng = np.random.default_rng(4)
    low_vol = 0.0005 + rng.normal(0, 0.0001, size=60)
    high_vol = 0.0005 + rng.normal(0, 0.001, size=60)
    returns = np.concatenate([low_vol, high_vol])
    features = _make_features(returns)

    model = KalmanFilter(instrument="BTC-USD", process_noise=1e-4, observation_noise=1e-2, warmup_period=10)
    model.fit(features.iloc[:20])
    signals = model.predict(features.iloc[20:])
    uncertainties = np.array([signal.uncertainty for signal in signals])

    mid = len(uncertainties) // 2
    assert uncertainties[mid:].mean() > uncertainties[:mid].mean()


def test_constant_returns_yield_zero_trend() -> None:
    returns = np.zeros(50)
    features = _make_features(returns)

    model = KalmanFilter(instrument="BTC-USD", process_noise=1e-6, observation_noise=1e-4, warmup_period=5)
    model.fit(features.iloc[:10])
    signals = model.predict(features.iloc[10:])
    scores = np.array([signal.score for signal in signals])

    assert np.allclose(scores, 0.0, atol=1e-8)


def test_get_uncertainty_matches_predict_without_state_change() -> None:
    returns = np.full(30, 0.001)
    features = _make_features(returns)

    model = KalmanFilter(instrument="BTC-USD", process_noise=1e-4, observation_noise=1e-2, warmup_period=5)
    model.fit(features.iloc[:10])

    future_features = features.iloc[10:]
    uncertainties_preview = model.get_uncertainty(future_features)
    signals = model.predict(future_features)
    uncertainties_actual = np.array([signal.uncertainty for signal in signals])

    assert np.allclose(uncertainties_preview, uncertainties_actual)
    # Ensure no NaN or negative variances surfaced.
    assert np.isfinite(uncertainties_actual).all()
    assert (uncertainties_actual >= 0).all()
