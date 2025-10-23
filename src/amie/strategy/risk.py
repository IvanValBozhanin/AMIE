"""Risk controls for the trading policy."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces position and drawdown limits for the strategy."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        risk_config = self._extract_risk_config(config)
        self.max_position_size = float(risk_config.get("max_position_size", 1.0))
        self.max_drawdown_pct = float(risk_config.get("max_drawdown_pct", 0.2))

        if self.max_position_size <= 0:
            raise ValueError("max_position_size must be positive")
        if self.max_drawdown_pct < 0:
            raise ValueError("max_drawdown_pct cannot be negative")

        self.current_drawdown_pct: float = 0.0

    @staticmethod
    def _extract_risk_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """Return the mapping that stores risk-specific settings."""
        if not config:
            return {}
        if "risk" in config and isinstance(config["risk"], Mapping):
            return config["risk"]
        return config

    def update_drawdown(self, current_equity: float, peak_equity: float) -> None:
        """Update drawdown state given current and peak equity levels."""
        if peak_equity <= 0:
            self.current_drawdown_pct = 0.0
            logger.warning(
                "Received non-positive peak equity %.4f; resetting drawdown state",
                peak_equity,
            )
            return

        drawdown = max(0.0, (peak_equity - current_equity) / peak_equity)
        self.current_drawdown_pct = drawdown

        if drawdown > self.max_drawdown_pct:
            logger.warning(
                "Drawdown %.2f%% exceeds maximum allowed %.2f%% (current_equity=%.4f, peak_equity=%.4f)",
                drawdown * 100,
                self.max_drawdown_pct * 100,
                current_equity,
                peak_equity,
            )
        else:
            logger.debug(
                "Drawdown updated to %.2f%% (current_equity=%.4f, peak_equity=%.4f)",
                drawdown * 100,
                current_equity,
                peak_equity,
            )

    def check_position(self, proposed_position: float, current_equity: float) -> float:
        """Return the risk-adjusted position size."""
        if self.current_drawdown_pct > self.max_drawdown_pct:
            logger.warning(
                "Drawdown %.2f%% above limit %.2f%%; forcing flat position "
                "(proposed_position=%.4f, equity=%.4f)",
                self.current_drawdown_pct * 100,
                self.max_drawdown_pct * 100,
                proposed_position,
                current_equity,
            )
            return 0.0

        capped_position = max(
            -self.max_position_size,
            min(self.max_position_size, proposed_position),
        )

        if capped_position != proposed_position:
            logger.info(
                "Position capped from %.4f to %.4f (max_position_size=%.4f, equity=%.4f)",
                proposed_position,
                capped_position,
                self.max_position_size,
                current_equity,
            )
        else:
            logger.debug(
                "Position accepted at %.4f (equity=%.4f)", capped_position, current_equity
            )

        return capped_position
