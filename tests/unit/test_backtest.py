from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from amie.backtest.engine import BacktestEngine
from amie.core.types import Signal
from amie.strategy.execution import ExecutionSimulator
from amie.strategy.policy import SignalPolicy
from amie.strategy.risk import RiskManager


def _make_features(
    periods: int,
    *,
    returns_value: float,
    price_start: float = 100.0,
    spread: float = 0.1,
) -> pd.DataFrame:
    ts_index = pd.date_range("2024-01-01", periods=periods, freq="min")
    prices = price_start + np.linspace(0.0, 1.0, periods)
    return pd.DataFrame(
        {
            "ts": ts_index,
            "instrument": ["TEST"] * periods,
            "returns": np.full(periods, returns_value, dtype=float),
            "price": prices,
            "spread": np.full(periods, spread, dtype=float),
        }
    )


@dataclass
class StubModel:
    """Simple stub that returns deterministic signals."""

    score_fn: Callable[[pd.Series], float]
    uncertainty: float = 1.0

    def predict(self, features: pd.DataFrame) -> list[Signal]:
        return [
            Signal(
                ts=row.ts,
                instrument=row.instrument,
                score=float(self.score_fn(row)),
                uncertainty=self.uncertainty,
                model_version="stub",
            )
            for row in features.itertuples(index=False)
        ]


def _construct_engine(
    model: StubModel,
    *,
    policy_cfg: dict | None = None,
    risk_cfg: dict | None = None,
    execution_cfg: dict | None = None,
) -> BacktestEngine:
    policy = SignalPolicy(policy_cfg or {"threshold_multiplier": 0.5, "max_position_size": 2.0})
    risk_manager = RiskManager(risk_cfg or {"max_position_size": 2.0, "max_drawdown_pct": 0.5})
    executor = ExecutionSimulator(execution_cfg or {"instrument": "TEST"})
    return BacktestEngine(
        {"initial_capital": 100_000.0},
        model=model,
        policy=policy,
        risk_manager=risk_manager,
        executor=executor,
    )


def test_positive_returns_increase_equity_and_respect_limits() -> None:
    features = _make_features(100, returns_value=0.002)
    model = StubModel(score_fn=lambda row: 3.0, uncertainty=0.8)
    engine = _construct_engine(
        model,
        policy_cfg={"threshold_multiplier": 1.0, "max_position_size": 1.5},
        risk_cfg={"max_position_size": 1.2, "max_drawdown_pct": 0.5},
        execution_cfg={"instrument": "TEST", "slippage_bps": 0.0, "fee_bps": 0.0},
    )

    result = engine.run(features)

    assert len(result) == len(features)
    assert (result["equity"].diff().fillna(0.0) >= -1e-9).all()
    assert result["equity"].iloc[-1] > 100_000.0
    assert (result["position"].abs() <= 1.2 + 1e-9).all()


def test_zero_signals_produce_no_trades_or_pnl() -> None:
    features = _make_features(50, returns_value=0.001)
    model = StubModel(score_fn=lambda row: 0.0, uncertainty=1.0)
    engine = _construct_engine(
        model,
        policy_cfg={"threshold_multiplier": 1.5, "max_position_size": 1.0},
        execution_cfg={"instrument": "TEST", "slippage_bps": 0.0, "fee_bps": 0.0},
    )

    result = engine.run(features)

    assert (result["position"] == 0.0).all()
    assert np.allclose(result["pnl"].to_numpy(), 0.0)
    assert np.allclose(result["equity"].to_numpy(), 100_000.0)


def test_zero_returns_result_in_fee_only_loss() -> None:
    features = _make_features(20, returns_value=0.0, spread=0.0)
    model = StubModel(score_fn=lambda row: 3.0, uncertainty=1.0)
    fee_bps = 5.0
    engine = _construct_engine(
        model,
        policy_cfg={"threshold_multiplier": 0.5, "max_position_size": 1.0},
        risk_cfg={"max_position_size": 1.0, "max_drawdown_pct": 0.5},
        execution_cfg={"instrument": "TEST", "slippage_bps": 0.0, "fee_bps": fee_bps},
    )

    result = engine.run(features)

    expected_fee = (fee_bps / 10_000.0) * features.loc[0, "price"] * 1.0
    assert pytest.approx(result["pnl"].sum(), rel=1e-6) == -expected_fee
    assert result["equity"].iloc[-1] == pytest.approx(100_000.0 - expected_fee)


def test_drawdown_does_not_exceed_limit() -> None:
    features = _make_features(30, returns_value=0.001)
    model = StubModel(score_fn=lambda row: 2.5, uncertainty=1.0)
    engine = _construct_engine(
        model,
        policy_cfg={"threshold_multiplier": 1.0, "max_position_size": 1.0},
        risk_cfg={"max_position_size": 1.0, "max_drawdown_pct": 0.3},
        execution_cfg={"instrument": "TEST", "slippage_bps": 0.0, "fee_bps": 0.0},
    )

    result = engine.run(features)

    assert (result["drawdown"] <= 0.3 + 1e-9).all()
