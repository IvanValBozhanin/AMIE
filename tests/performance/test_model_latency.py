from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amie.models.kalman import KalmanFilter


def _make_features(n: int) -> pd.DataFrame:
    returns = np.linspace(0.0, 0.001, n) + np.random.default_rng(0).normal(0, 0.0005, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="s")
    instrument = ["BTC-USD"] * n
    zeros = np.zeros(n)
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


@pytest.mark.benchmark(group="kalman")
def test_kalman_update_single_tick_latency(benchmark) -> None:
    model = KalmanFilter(instrument="BTC-USD", process_noise=1e-4, observation_noise=1e-2, warmup_period=5)
    features = _make_features(10)
    model.fit(features)

    single_tick = features.iloc[9:10]

    def _predict_single():
        model.predict(single_tick)

    benchmark(_predict_single)
    assert benchmark.stats.stats.mean < 0.001  # 1 ms budget


@pytest.mark.benchmark(group="kalman")
def test_kalman_predict_100_latency(benchmark) -> None:
    model = KalmanFilter(instrument="BTC-USD", process_noise=1e-4, observation_noise=1e-2, warmup_period=5)
    features = _make_features(150)
    model.fit(features.iloc[:50])
    batch = features.iloc[50:150]

    def _predict():
        return model.predict(batch)

    result = benchmark(_predict)
    assert benchmark.stats.stats.mean < 0.050  # 50 ms budget
    assert len(result) == len(batch)


@pytest.mark.benchmark(group="kalman")
def test_kalman_predict_1000_latency(benchmark) -> None:
    model = KalmanFilter(instrument="BTC-USD", process_noise=1e-4, observation_noise=1e-2, warmup_period=5)
    features = _make_features(1_200)
    model.fit(features.iloc[:200])
    batch = features.iloc[200:]

    def _predict():
        return model.predict(batch)

    result = benchmark(_predict)
    assert benchmark.stats.stats.mean < 0.200  # 200 ms budget
    assert len(result) == len(batch)
