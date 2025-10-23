"""Performance metrics for backtest results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "compute_sharpe",
    "compute_sortino",
    "compute_max_drawdown",
    "compute_calmar",
    "compute_hit_rate",
    "compute_turnover",
    "BacktestMetrics",
]


def _clean_array(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    mask = ~np.isnan(arr)
    return arr[mask]


def compute_sharpe(returns: np.ndarray, periods_per_year: float = 252) -> float:
    """Return the annualised Sharpe ratio for the supplied returns."""
    cleaned = _clean_array(returns)
    if cleaned.size == 0:
        return 0.0

    mean_return = cleaned.mean()
    volatility = cleaned.std(ddof=1) if cleaned.size > 1 else cleaned.std(ddof=0)

    if volatility == 0:
        return 0.0

    scale = np.sqrt(periods_per_year) if periods_per_year > 0 else 1.0
    return float((mean_return * periods_per_year) / (volatility * scale))


def compute_sortino(returns: np.ndarray, target: float = 0.0, periods_per_year: float = 252) -> float:
    """Return the Sortino ratio for the supplied returns."""
    cleaned = _clean_array(returns)
    if cleaned.size == 0:
        return 0.0

    downside_diff = cleaned - target
    downside = downside_diff[downside_diff < 0]

    if downside.size == 0:
        return float("inf")

    downside_deviation = np.sqrt(np.mean(np.square(downside)))
    if downside_deviation == 0:
        return float("inf")

    mean_return = cleaned.mean()
    return float((mean_return - target) * periods_per_year / (downside_deviation * np.sqrt(periods_per_year)))


def compute_max_drawdown(equity: np.ndarray) -> float:
    """Return the maximum drawdown expressed as a fraction."""
    cleaned = _clean_array(equity)
    if cleaned.size == 0:
        return 0.0

    peak = np.maximum.accumulate(cleaned)
    drawdowns = (peak - cleaned) / peak
    return float(np.max(drawdowns))


def compute_calmar(returns: np.ndarray, max_drawdown: float, periods_per_year: float = 252) -> float:
    """Return the Calmar ratio given returns and maximum drawdown."""
    cleaned = _clean_array(returns)
    if cleaned.size == 0:
        return 0.0

    annual_return = cleaned.mean() * periods_per_year
    if max_drawdown <= 0:
        return float("inf") if annual_return > 0 else 0.0

    return float(annual_return / max_drawdown)


def compute_hit_rate(returns: np.ndarray) -> float:
    """Return the proportion of positive returns."""
    cleaned = _clean_array(returns)
    if cleaned.size == 0:
        return 0.0

    positives = np.count_nonzero(cleaned > 0)
    return float(positives / cleaned.size)


def compute_turnover(positions: np.ndarray) -> float:
    """Return the absolute turnover based on position changes."""
    cleaned = _clean_array(positions)
    if cleaned.size < 2:
        return 0.0
    diffs = np.diff(cleaned)
    return float(np.nansum(np.abs(diffs)))


class BacktestMetrics:
    """Convenience wrapper that computes a standard metric set."""

    def __init__(self, results_df: pd.DataFrame, *, periods_per_year: float = 252) -> None:
        if results_df is None or results_df.empty:
            raise ValueError("results_df must be a populated DataFrame")

        self.df = results_df.copy()
        self.periods_per_year = periods_per_year

        returns = self._extract_series(["returns", "pnl"], default=0.0)
        equity = self._extract_series(["equity"], default=np.nan)
        positions = self._extract_series(["position"], default=0.0)

        self.returns = returns.to_numpy(dtype=float)
        self.equity = equity.to_numpy(dtype=float) if not equity.isna().all() else np.cumsum(self.returns)
        self.positions = positions.to_numpy(dtype=float)

        self.metrics: dict[str, Any] = self._compute_metrics()

    def _extract_series(self, candidates: list[str], default: float) -> pd.Series:
        for column in candidates:
            if column in self.df.columns:
                return pd.to_numeric(self.df[column], errors="coerce")
        return pd.Series([default] * len(self.df), dtype=float)

    def _compute_metrics(self) -> dict[str, Any]:
        max_drawdown = compute_max_drawdown(self.equity)

        metrics = {
            "sharpe": compute_sharpe(self.returns, self.periods_per_year),
            "sortino": compute_sortino(self.returns, periods_per_year=self.periods_per_year),
            "max_drawdown": max_drawdown,
            "calmar": compute_calmar(self.returns, max_drawdown, periods_per_year=self.periods_per_year),
            "hit_rate": compute_hit_rate(self.returns),
            "turnover": compute_turnover(self.positions),
        }
        return metrics

    def to_dict(self) -> dict[str, Any]:
        return {key: (None if value is None or (isinstance(value, float) and np.isnan(value)) else value) for key, value in self.metrics.items()}

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
