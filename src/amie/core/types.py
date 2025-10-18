"""Typed entities used across the AMIE pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass

from .constants import Side

__all__ = [
    "Tick",
    "LOBSnapshot",
    "FeatureVector",
    "Signal",
    "Decision",
    "Fill",
    "RunMetadata",
]


class SerializableDataclass:
    """Mixin that adds dict/json helpers backed by Pydantic's TypeAdapter."""

    _type_adapter_cache: ClassVar[dict[type["SerializableDataclass"], TypeAdapter[Any]]] = {}

    @classmethod
    def _type_adapter(cls) -> TypeAdapter[Any]:
        """Return the cached type adapter for the current dataclass."""
        adapter = cls._type_adapter_cache.get(cls)
        if adapter is None:
            adapter = TypeAdapter(cls)
            cls._type_adapter_cache[cls] = adapter
        return adapter

    def dict(self) -> dict[str, Any]:
        """Return the dataclass as a Python dictionary."""
        return self._type_adapter().dump_python(self, mode="python")

    def json(self, *, indent: int | None = None) -> str:
        """Return the dataclass as a JSON string."""
        return self._type_adapter().dump_json(self, indent=indent).decode()


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class Tick(SerializableDataclass):
    """Individual market data tick as observed in the feed."""

    ts: datetime
    instrument: str
    price: float
    qty: float
    side: Side


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class LOBSnapshot(SerializableDataclass):
    """Level-2 snapshot of the limit order book for an instrument."""

    ts: datetime
    instrument: str
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    spread: float


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class FeatureVector(SerializableDataclass):
    """Feature representation derived from market data."""

    ts: datetime
    instrument: str
    returns: float
    volatility: float
    z_score: float
    spread: float
    imbalance: float


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class Signal(SerializableDataclass):
    """Model output describing directional conviction."""

    ts: datetime
    instrument: str
    score: float
    uncertainty: float
    model_version: str


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class Decision(SerializableDataclass):
    """Trading decision generated from a signal."""

    ts: datetime
    instrument: str
    target_position: float
    rationale: str


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class Fill(SerializableDataclass):
    """Execution fill information returned from the broker."""

    ts: datetime
    instrument: str
    qty: float
    price: float
    slippage: float


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class RunMetadata(SerializableDataclass):
    """Metadata describing the runtime context for a strategy invocation."""

    run_id: str
    seed: int
    config_hash: str
    git_sha: str
    timestamp: datetime
