"""Backtest engine that ties together the model, policy, risk, and execution."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from amie.core.types import Signal
from amie.strategy.execution import ExecutionSimulator
from amie.strategy.policy import SignalPolicy
from amie.strategy.risk import RiskManager
from amie.utils.profiling import profile

logger = logging.getLogger(__name__)

__all__ = ["BacktestEngine"]


class BacktestEngine:
    """Vectorised single-asset backtest loop."""

    def __init__(
        self,
        config: Mapping[str, Any] | None,
        model: Any,
        policy: SignalPolicy,
        risk_manager: RiskManager,
        executor: ExecutionSimulator,
    ) -> None:
        self.config = config or {}
        self.model = model
        self.policy = policy
        self.risk_manager = risk_manager
        self.executor = executor

        self.initial_capital = float(self.config.get("initial_capital", 100_000.0))

    @profile("BacktestEngine.run")
    def run(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Run the backtest over the provided feature frame."""
        df = self._prepare_features(features_df)
        signals = self._predict_signals(df)

        position = 0.0
        equity = self.initial_capital
        peak_equity = equity
        records: list[dict[str, Any]] = []
        total_trades = 0

        for idx, row in enumerate(df.itertuples(index=False)):
            signal = signals[idx]
            target_position = float(self.policy.compute_position(signal))
            risk_adjusted_position = float(
                self.risk_manager.check_position(target_position, equity)
            )

            order_qty = risk_adjusted_position - position
            trade_cost = 0.0

            if abs(order_qty) > 1e-12:
                fill = self.executor.execute(
                    order_qty=order_qty,
                    market_price=getattr(row, "price"),
                    spread=getattr(row, "spread"),
                )
                fee = getattr(self.executor, "last_fee", 0.0)
                trade_cost = abs(fill.slippage) * abs(fill.qty) + fee
                total_trades += 1

            pnl = risk_adjusted_position * float(getattr(row, "returns")) - trade_cost
            equity += pnl
            peak_equity = max(peak_equity, equity)

            self.risk_manager.update_drawdown(equity, peak_equity)
            drawdown = self.risk_manager.current_drawdown_pct

            records.append(
                {
                    "ts": getattr(row, "ts"),
                    "position": risk_adjusted_position,
                    "pnl": pnl,
                    "equity": equity,
                    "drawdown": drawdown,
                    "signal_score": signal.score,
                    "uncertainty": signal.uncertainty,
                }
            )

            position = risk_adjusted_position

        result = pd.DataFrame(records)
        sharpe = self._compute_sharpe(result["pnl"].to_numpy() if not result.empty else np.array([]))

        logger.info(
            "backtest_summary total_trades=%d final_equity=%.2f sharpe=%.4f",
            total_trades,
            float(equity),
            sharpe,
        )
        return result

    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if features_df is None or features_df.empty:
            raise ValueError("features_df must be a non-empty DataFrame")

        df = features_df.copy()
        required_columns = {"ts", "returns", "price", "spread"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Features DataFrame missing required columns: {sorted(missing)}")

        df.sort_values("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _predict_signals(self, df: pd.DataFrame) -> Sequence[Signal]:
        predicted = self.model.predict(df)
        signals = list(predicted)
        if len(signals) != len(df):
            raise ValueError(
                f"Model returned {len(signals)} signals for {len(df)} feature rows"
            )
        return signals

    def _compute_sharpe(self, pnl: np.ndarray) -> float:
        if pnl.size == 0:
            return float("nan")
        volatility = pnl.std(ddof=1) if pnl.size > 1 else 0.0
        if volatility > 0:
            return float(pnl.mean() / volatility * np.sqrt(pnl.size))
        return 0.0
