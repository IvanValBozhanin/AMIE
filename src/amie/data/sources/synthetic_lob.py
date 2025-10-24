# CONTRACT (Module)
# Purpose:
#   Produce deterministic synthetic limit order book regimes and export ticks/LOB snapshots for testing.
# Public API:
#   - SyntheticLOBGenerator(seed:int, config:SyntheticLOBConfig|None)
#   - SyntheticLOBGenerator.generate(num_ticks:int=10_000) -> Iterator[tuple[Tick, LOBSnapshot]]
#   - SyntheticLOBGenerator.to_dataframe(num_ticks:int) -> pd.DataFrame
# Inputs:
#   - seed: integer forwarded to np.random.default_rng for reproducible draws each generate call.
#   - config: SyntheticLOBConfig with regimes (volatility, spread, qty), regime_duration, depth, level_spread,
#     tick_interval_seconds, instrument.
#   - num_ticks: int >= 0 controlling length of stream/DataFrame; 0 yields empty iterator/DataFrame.
# Outputs:
#   - generate: yields (Tick, LOBSnapshot) pairs with matching timestamps and instrument per tick.
#   - to_dataframe: DataFrame with columns ['ts','instrument','price','qty','side','spread','bid_price','bid_qty',
#     'ask_price','ask_qty'] ordered by ts and length == num_ticks.
# Invariants:
#   - For fixed seed/config the tick stream is deterministic; RNG re-initialised per generate() call.
#   - Timestamps start at 2024-01-01 00:00:00 and advance by tick_interval_seconds; strictly increasing.
#   - Price and qty lower-bounded by 1e-6; spread>=0 as supplied by regime; bids < asks at each depth level.
#   - Bids/asks lists each have length == config.depth with non-negative quantities; LOBSnapshot spreads echo regime.
#   - Tick.side sampled from Side enum; recorded as Side.value preserving deterministic order.
# TODO:
#   - Confirm Side enum values match downstream schema expectations (string vs integer codes).

"""Synthetic limit order book generator for testing and research."""

from __future__ import annotations

from dataclasses import field
from datetime import datetime, timedelta
from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

from amie.core.constants import Side
from amie.core.types import LOBSnapshot, Tick

__all__ = [
    "SyntheticLOBRegime",
    "SyntheticLOBConfig",
    "SyntheticLOBGenerator",
]


@dataclass(config={"frozen": True})
class SyntheticLOBRegime:
    """Configuration parameters for a single synthetic regime."""

    volatility: float = 0.0005
    spread: float = 25.0
    qty: float = 1.0


def _default_regimes() -> list[SyntheticLOBRegime]:
    return [
        SyntheticLOBRegime(volatility=0.0004, spread=20.0, qty=1.0),
        SyntheticLOBRegime(volatility=0.0008, spread=35.0, qty=1.2),
    ]


@dataclass(config={"frozen": True})
class SyntheticLOBConfig:
    """Configuration controlling the synthetic generator behaviour."""

    instrument: str = "BTC-USD"
    regimes: Sequence[SyntheticLOBRegime] = field(default_factory=_default_regimes)
    regime_duration: int = 250
    depth: int = 3
    level_spread: float = 10.0
    tick_interval_seconds: float = 1.0


class SyntheticLOBGenerator:
    """Deterministic synthetic limit order book tick stream generator."""

    base_price: float = 50_000.0

    def __init__(self, seed: int, config: SyntheticLOBConfig | None = None) -> None:
        self._seed = seed
        self.config = config or SyntheticLOBConfig()
        self._start_timestamp = datetime.fromisoformat("2024-01-01T00:00:00")

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self._seed)

    def _regime_for_tick(self, tick_index: int) -> SyntheticLOBRegime:
        regime_idx = (tick_index // self.config.regime_duration) % len(self.config.regimes)
        return self.config.regimes[regime_idx]

    def _side(self, rng: np.random.Generator) -> Side:
        return Side(rng.choice([side.value for side in Side]))

    def _book_levels(
        self,
        price: float,
        spread: float,
        qty: float,
    ) -> Tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        bids: list[tuple[float, float]] = []
        asks: list[tuple[float, float]] = []
        for level in range(self.config.depth):
            level_factor = max(0.2, 1.0 - 0.15 * level)
            level_qty = max(qty * level_factor, 1e-6)
            price_offset = self.config.level_spread * level
            bids.append((price - spread / 2 - price_offset, level_qty))
            asks.append((price + spread / 2 + price_offset, level_qty))
        return bids, asks

    def generate(self, num_ticks: int = 10_000) -> Iterator[tuple[Tick, LOBSnapshot]]:
        """Yield a deterministic stream of ticks and LOB snapshots."""
        # MINI-CONTRACT: SyntheticLOBGenerator.generate
        # Inputs:
        #   num_ticks: integer >= 0 specifying number of (Tick, LOBSnapshot) pairs to emit.
        # Outputs:
        #   Iterator producing num_ticks sequential pairs with shared timestamp/instrument.
        # Invariants:
        #   - Uses fresh RNG seeded with self._seed => repeated calls reproduce identical sequence.
        #   - Timestamps strictly increasing; price, qty >= 1e-6; regime cycles every regime_duration ticks.
        #   - Bids/asks lengths equal config.depth; spread equals regime.spread each tick.
        rng = self._rng()
        price = self.base_price
        for idx in range(num_ticks):
            regime = self._regime_for_tick(idx)
            price_change = rng.normal(0.0, regime.volatility * price)
            price = max(price + price_change, 1e-6)
            qty = max(rng.normal(regime.qty, regime.qty * 0.1), 1e-6)
            spread = regime.spread
            bids, asks = self._book_levels(price, spread, qty)
            ts = self._start_timestamp + timedelta(seconds=idx * self.config.tick_interval_seconds)

            tick = Tick(
                ts=ts,
                instrument=self.config.instrument,
                price=price,
                qty=qty,
                side=self._side(rng),
            )
            lob = LOBSnapshot(
                ts=ts,
                instrument=self.config.instrument,
                bids=bids,
                asks=asks,
                spread=spread,
            )
            yield tick, lob

    def to_dataframe(self, num_ticks: int) -> pd.DataFrame:
        """Generate ticks and convert the stream to a pandas DataFrame."""
        # MINI-CONTRACT: SyntheticLOBGenerator.to_dataframe
        # Inputs:
        #   num_ticks: integer >= 0 forwarded to generate().
        # Outputs:
        #   DataFrame length num_ticks with columns ['ts','instrument','price','qty','side','spread','bid_price',
        #   'bid_qty','ask_price','ask_qty'] sorted by ts.
        # Invariants:
        #   - Delegates to generate(), so deterministic under fixed seed/config.
        #   - Best bid/ask extracted as first level when available; NaN if depth==0.
        #   - Does not mutate generator state beyond deterministic consumption.
        records = []
        for tick, lob in self.generate(num_ticks):
            best_bid = lob.bids[0] if lob.bids else (np.nan, np.nan)
            best_ask = lob.asks[0] if lob.asks else (np.nan, np.nan)
            records.append(
                {
                    "ts": tick.ts,
                    "instrument": tick.instrument,
                    "price": tick.price,
                    "qty": tick.qty,
                    "side": tick.side.value,
                    "spread": lob.spread,
                    "bid_price": best_bid[0],
                    "bid_qty": best_bid[1],
                    "ask_price": best_ask[0],
                    "ask_qty": best_ask[1],
                }
            )
        return pd.DataFrame.from_records(records)
