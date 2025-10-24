# CONTRACT (Module)
# Purpose:
#   Orchestrate single-asset backtests by wiring model signals through policy, risk, and execution layers.
# Public API:
#   - BacktestEngine(config, model, policy, risk_manager, executor)
#   - BacktestEngine.run(features_df: pd.DataFrame) -> pd.DataFrame
# Inputs:
#   - features_df: DataFrame with columns {'ts','returns','price','spread'}; any index; values numeric.
#   - Dependencies: model.predict(DataFrame) -> Sequence[Signal]; policy.compute_position(Signal) -> float;
#     risk_manager.check_position(float, equity) -> float and update_drawdown(equity, peak_equity); executor.execute(...)
#     returns fill with attrs qty, slippage and exposes last_fee.
# Outputs:
#   - Backtest result DataFrame with columns ['ts','position','pnl','equity','drawdown','signal_score','uncertainty']
#     aligned 1:1 with sorted input feature rows.
# Invariants:
#   - features_df copied, sorted by 'ts', index reset; length preserved; positional loop uses chronological order.
#   - Signals length must equal features (validated); each row uses current signal, prior position, and equity only.
#   - order_qty = risk_adjusted_position - previous position; executor called only when |order_qty|>1e-12.
#   - pnl = risk_adjusted_position * returns - trade_cost where trade_cost >=0 assuming executor fees/slippage >=0.
#   - equity evolves deterministically from initial_capital + cumulative pnl; drawdown sourced from risk_manager.
#   - Result columns contain finite floats unless upstream dependencies return NaNs; logging emits summary stats.
# TODO:
#   - Confirm monetary units (returns fractional vs bps) and fee conventions for accurate pnl interpretation.

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
        # MINI-CONTRACT: BacktestEngine.run
        # Inputs:
        #   features_df: DataFrame with {'ts','returns','price','spread'}; may contain extra columns.
        # Outputs:
        #   DataFrame sorted by ts with columns ['ts','position','pnl','equity','drawdown','signal_score','uncertainty'];
        #   len == len(features_df); equity starts at initial_capital and evolves via pnl recursion.
        # Invariants:
        #   - Consumes model.predict(df) sequentially; signals must align with features 1:1.
        #   - Applies policy/risk/executor per row with no look-ahead beyond current equity/position.
        #   - Trade cost only applied when |order_qty|>1e-12; total_trades counts such events.
        #   - Returns are NaN-safe: if returns[i] is NaN => pnl NaN propagates to equity/drawdown for that step.
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

    # MINI-CONTRACT: BacktestEngine._prepare_features
    # Inputs:
    #   features_df: non-empty DataFrame expected to contain {'ts','returns','price','spread'}.
    # Outputs:
    #   Sorted copy with RangeIndex; caller's DataFrame left untouched.
    # Invariants:
    #   - Raises ValueError on missing required columns or empty frame.
    #   - Sorting enforces chronological order for downstream sequential loop.
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

    # MINI-CONTRACT: BacktestEngine._predict_signals
    # Inputs:
    #   df: Prepared features DataFrame already sorted and index-reset.
    # Outputs:
    #   list[Signal] aligned 1:1 with df rows.
    # Invariants:
    #   - Raises ValueError if model output length mismatch detected.
    #   - Preserves df order when materialising iterator from model.
    def _predict_signals(self, df: pd.DataFrame) -> Sequence[Signal]:
        predicted = self.model.predict(df)
        signals = list(predicted)
        if len(signals) != len(df):
            raise ValueError(
                f"Model returned {len(signals)} signals for {len(df)} feature rows"
            )
        return signals

    def _compute_sharpe(self, pnl: np.ndarray) -> float:
        # MINI-CONTRACT: BacktestEngine._compute_sharpe
        # Inputs:
        #   pnl: np.ndarray of per-period pnl values (may be empty).
        # Outputs:
        #   Annualised Sharpe ratio float; NaN if no samples, 0.0 if volatility == 0.
        # Invariants:
        #   - Uses ddof=1 when >=2 samples; guards against divide-by-zero via early returns.
        if pnl.size == 0:
            return float("nan")
        volatility = pnl.std(ddof=1) if pnl.size > 1 else 0.0
        if volatility > 0:
            return float(pnl.mean() / volatility * np.sqrt(pnl.size))
        return 0.0
