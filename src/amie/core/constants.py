"""Enumerations and default constants for AMIE trading workflows."""

from enum import Enum

__all__ = [
    "Side",
    "Position",
    "MAX_POSITION_SIZE",
    "SLIPPAGE_BPS",
    "DEFAULT_TRADE_NOTIONAL",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_TARGET_SPREAD",
]


class Side(str, Enum):
    """Order direction supported by the execution layer."""

    BUY = "BUY"
    SELL = "SELL"


class Position(str, Enum):
    """Gross position state for an instrument."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


# Default parameter values for strategy configuration.
MAX_POSITION_SIZE: float = 1.0
SLIPPAGE_BPS: int = 5
DEFAULT_TRADE_NOTIONAL: float = 1_000.0
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6
DEFAULT_TARGET_SPREAD: float = 0.5
