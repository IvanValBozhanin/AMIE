"""Simple execution simulator that applies deterministic slippage and fees."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from amie.core.types import Fill

logger = logging.getLogger(__name__)


class ExecutionSimulator:
    """Apply fixed slippage and fee assumptions to derive fills."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        execution_config = self._extract_execution_config(config)
        self.slippage_bps = float(execution_config.get("slippage_bps", 0.0))
        self.fee_bps = float(execution_config.get("fee_bps", 0.0))
        self.instrument = execution_config.get("instrument", "UNKNOWN")

        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")
        if self.fee_bps < 0:
            raise ValueError("fee_bps cannot be negative")

        self.last_fee: float = 0.0

    @staticmethod
    def _extract_execution_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if not config:
            return {}
        if "execution" in config and isinstance(config["execution"], Mapping):
            return config["execution"]
        return config

    def execute(self, order_qty: float, market_price: float, spread: float) -> Fill:
        """Return the simulated fill for a submitted order."""
        if market_price <= 0:
            raise ValueError("market_price must be positive")
        if spread < 0:
            raise ValueError("spread cannot be negative")

        side = 1.0 if order_qty > 0 else -1.0 if order_qty < 0 else 0.0
        slippage = self._compute_slippage(market_price, spread)

        if side > 0:
            executed_price = market_price + slippage
            realized_slippage = slippage
        elif side < 0:
            executed_price = market_price - slippage
            realized_slippage = -slippage
        else:
            executed_price = market_price
            realized_slippage = 0.0

        self.last_fee = self._compute_fee(order_qty, market_price)

        logger.debug(
            "ExecutionSimulator fill: qty=%.4f market_price=%.4f spread=%.4f "
            "slippage=%.4f fee=%.4f executed_price=%.4f",
            order_qty,
            market_price,
            spread,
            realized_slippage,
            self.last_fee,
            executed_price,
        )

        return Fill(
            ts=datetime.now(UTC),
            instrument=self.instrument,
            qty=order_qty,
            price=executed_price,
            slippage=realized_slippage,
        )

    def _compute_slippage(self, market_price: float, spread: float) -> float:
        """Return absolute slippage applied to executions."""
        bps_component = (self.slippage_bps / 10_000.0) * market_price
        spread_component = spread / 2.0
        return bps_component + spread_component

    def _compute_fee(self, order_qty: float, market_price: float) -> float:
        """Return total fee in currency units for the trade."""
        notional = abs(order_qty) * market_price
        return (self.fee_bps / 10_000.0) * notional
