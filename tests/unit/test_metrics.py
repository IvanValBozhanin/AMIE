from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amie.backtest.metrics import (
    BacktestMetrics,
    compute_calmar,
    compute_hit_rate,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_turnover,
)


def test_compute_sharpe_expected_value() -> None:
    returns = np.array([0.03, -0.01, 0.01, -0.01, 0.03])
    result = compute_sharpe(returns, periods_per_year=1)
    assert result == pytest.approx(0.5, abs=1e-2)


def test_compute_max_drawdown_known_curve() -> None:
    equity = np.array([100, 110, 105, 120, 90, 95])
    result = compute_max_drawdown(equity)
    assert result == pytest.approx((120 - 90) / 120)


def test_compute_hit_rate_sixty_percent_positive() -> None:
    returns = np.array([0.01] * 6 + [-0.01] * 4)
    np.random.shuffle(returns)
    result = compute_hit_rate(returns)
    assert result == pytest.approx(0.6, abs=1e-6)


def test_zero_returns_yield_zero_sharpe() -> None:
    returns = np.zeros(10)
    result = compute_sharpe(returns, periods_per_year=1)
    assert result == pytest.approx(0.0)


def test_negative_returns_produce_negative_sharpe() -> None:
    returns = np.full(20, -0.01)
    result = compute_sharpe(returns, periods_per_year=1)
    assert result < 0


def test_edge_cases_single_return_and_nans() -> None:
    single = np.array([0.03])
    assert compute_sharpe(single, periods_per_year=1) == pytest.approx(0.0)

    nan_returns = np.array([np.nan, np.nan])
    assert compute_hit_rate(nan_returns) == pytest.approx(0.0)
    assert compute_turnover(nan_returns) == pytest.approx(0.0)
    assert compute_calmar(nan_returns, max_drawdown=0.1, periods_per_year=1) == pytest.approx(0.0)
    assert np.isinf(compute_sortino(np.array([0.01, 0.02, 0.03]), target=0, periods_per_year=1))


def test_backtest_metrics_end_to_end(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=5, freq="min"),
            "returns": [0.01, -0.005, 0.015, 0.0, 0.02],
            "equity": [100, 101, 100.5, 102, 102.5],
            "position": [0.0, 1.0, 1.0, 0.5, 0.5],
        }
    )

    metrics = BacktestMetrics(df, periods_per_year=1)
    metrics_dict = metrics.to_dict()

    assert "sharpe" in metrics_dict
    assert metrics_dict["turnover"] == pytest.approx(1.5)

    out_path = tmp_path / "metrics.json"
    metrics.save(out_path)

    saved = json.loads(out_path.read_text())
    assert saved == metrics_dict

    df_metrics = metrics.to_dataframe()
    assert df_metrics.shape == (1, len(metrics_dict))
