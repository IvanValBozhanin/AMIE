from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amie.features.transforms import FeatureComputer


def _make_df(n: int = 100, price: float = 50_000.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=n, freq="s")
    prices = price + rng.normal(0, price * 0.0005, size=n).cumsum()
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


def test_feature_computation_no_nan_after_warmup() -> None:
    df = _make_df(100)
    computer = FeatureComputer(window_size=20)
    features = computer.compute(df)

    warmup = computer.window_size
    assert features.iloc[warmup:].notna().all().all()


def test_volatility_zero_for_constant_prices() -> None:
    df = _make_df(100, price=50_000.0)
    df["price"] = 50_000.0
    df["bid_price"] = 49_990.0
    df["ask_price"] = 50_010.0

    computer = FeatureComputer(window_size=5)
    features = computer.compute(df)

    assert np.allclose(features["ewma_volatility"], 0.0)


def test_z_score_zero_when_returns_zero_variance() -> None:
    df = _make_df(100)
    df["price"] = np.linspace(50_000.0, 50_100.0, num=len(df))
    df["bid_price"] = df["price"] - 10
    df["ask_price"] = df["price"] + 10

    computer = FeatureComputer(window_size=10)
    features = computer.compute(df)
    zero_variance_mask = features["returns"].rolling(window=10, min_periods=1).std(ddof=0) == 0
    assert np.allclose(features.loc[zero_variance_mask, "z_score"], 0.0, equal_nan=True)


def test_z_score_symmetry() -> None:
    df = _make_df(100)
    # Symmetric returns around zero
    returns = np.concatenate([np.linspace(-0.01, 0.0, 50), np.linspace(0.0, 0.01, 50)])
    prices = 50_000.0 * np.exp(np.cumsum(returns))
    df["price"] = prices
    df["bid_price"] = df["price"] - 10
    df["ask_price"] = df["price"] + 10

    computer = FeatureComputer(window_size=20)
    features = computer.compute(df)

    z_scores = features["z_score"].dropna()
    assert np.isclose(z_scores.mean(), 0.0, atol=1e-3)


def test_single_row_returns_nan() -> None:
    df = _make_df(1)
    computer = FeatureComputer(window_size=5)
    features = computer.compute(df)

    assert features.isna().any().any()


def test_missing_columns_raises() -> None:
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, freq="s")})
    computer = FeatureComputer()
    with pytest.raises(ValueError):
        computer.compute(df)
