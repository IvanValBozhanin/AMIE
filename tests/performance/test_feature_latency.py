from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amie.features.store import FeatureStore
from amie.features.transforms import FeatureComputer


def _generate_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    times = pd.date_range("2024-01-01", periods=n, freq="s")
    prices = 50_000 + rng.normal(0, 50, size=n).cumsum()
    bid_prices = prices - 10
    ask_prices = prices + 10
    bid_qty = rng.uniform(0.5, 2.0, size=n)
    ask_qty = rng.uniform(0.5, 2.0, size=n)
    return pd.DataFrame(
        {
            "ts": times,
            "instrument": "BTC-USD",
            "price": prices,
            "bid_price": bid_prices,
            "ask_price": ask_prices,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
        }
    )


@pytest.mark.benchmark(group="features")
def test_feature_computation_latency_100(benchmark) -> None:
    df = _generate_df(100)
    fc = FeatureComputer()
    result = benchmark(fc.compute, df)
    assert benchmark.stats.stats.mean < 0.050  # 50 ms budget
    assert not result.empty


@pytest.mark.benchmark(group="features")
def test_feature_computation_latency_1000(benchmark) -> None:
    df = _generate_df(1_000)
    fc = FeatureComputer()
    result = benchmark(fc.compute, df)
    assert benchmark.stats.stats.mean < 0.200  # 200 ms budget
    assert not result.empty


@pytest.mark.benchmark(group="feature_store")
def test_parquet_write_latency(benchmark, tmp_path: Path) -> None:
    df = _generate_df(1_000)
    fc = FeatureComputer()
    features = fc.compute(df)
    store = FeatureStore(base_path=tmp_path / "features")

    def _write():
        store.write(features, run_id="bench-write")

    benchmark(_write)
    assert benchmark.stats.stats.mean < 0.100


@pytest.mark.benchmark(group="feature_store")
def test_parquet_read_latency(benchmark, tmp_path: Path) -> None:
    df = _generate_df(1_000)
    fc = FeatureComputer()
    features = fc.compute(df)
    store = FeatureStore(base_path=tmp_path / "features")
    store.write(features, run_id="bench-read")

    def _read():
        return store.read("bench-read")

    result = benchmark(_read)
    assert benchmark.stats.stats.mean < 0.050
    assert not result.empty
