"""Data validation utilities built on top of Pandera schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Field
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing import Series

from amie.core.constants import Side
from amie.core.types import FeatureVector, LOBSnapshot, Tick

__all__ = [
    "ValidationError",
    "TickSchema",
    "LOBSnapshotSchema",
    "FeatureVectorSchema",
    "validate_tick",
    "validate_lob",
    "validate_feature_vector",
]


class ValidationError(ValueError):
    """Raised when structured data fails validation."""


class TickSchema(pa.SchemaModel):
    """Schema describing a single tick record."""

    ts: Series[pd.Timestamp] = Field(coerce=True)
    instrument: Series[str] = Field(str_length={"min_value": 1})
    price: Series[float] = Field(gt=0.0)
    qty: Series[float] = Field(gt=0.0)
    side: Series[str] = Field(isin=[side.value for side in Side])

    class Config:
        strict = True
        coerce = True


class LOBSnapshotSchema(pa.SchemaModel):
    """Schema describing a limit order book snapshot summary."""

    ts: Series[pd.Timestamp] = Field(coerce=True)
    instrument: Series[str] = Field(str_length={"min_value": 1})
    spread: Series[float] = Field(ge=0.0)
    depth: Series[int] = Field(ge=0, coerce=True)

    class Config:
        strict = True
        coerce = True


class FeatureVectorSchema(pa.SchemaModel):
    """Schema describing engineered feature values."""

    ts: Series[pd.Timestamp] = Field(coerce=True)
    instrument: Series[str] = Field(str_length={"min_value": 1})
    returns: Series[float] = Field(nullable=False)
    volatility: Series[float] = Field(nullable=False)
    z_score: Series[float] = Field(nullable=False)
    spread: Series[float] = Field(nullable=False)
    imbalance: Series[float] = Field(nullable=False)

    class Config:
        strict = True
        coerce = True


def _to_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    return value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)


def _validate_record(
    schema: type[pa.SchemaModel],
    payload: Mapping[str, Any],
    entity_label: str,
) -> bool:
    df = pd.DataFrame([payload])
    try:
        schema.validate(df, lazy=True)
    except SchemaError as exc:
        raise ValidationError(f"{entity_label} validation failed: {exc}") from exc
    except SchemaErrors as exc:
        failure_cases = exc.failure_cases.to_dict(orient="records")
        raise ValidationError(f"{entity_label} validation failed: {failure_cases}") from exc
    return True


def validate_tick(tick: Tick) -> bool:
    """Validate a Tick instance and return True if it is valid."""
    record = {
        "ts": _to_timestamp(tick.ts),
        "instrument": tick.instrument,
        "price": float(tick.price),
        "qty": float(tick.qty),
        "side": tick.side.value if isinstance(tick.side, Side) else str(tick.side),
    }
    return _validate_record(TickSchema, record, "Tick")


def validate_lob(lob: LOBSnapshot) -> bool:
    """Validate a LOBSnapshot instance and return True if it is valid."""
    depth = max(len(lob.bids), len(lob.asks))
    record = {
        "ts": _to_timestamp(lob.ts),
        "instrument": lob.instrument,
        "spread": float(lob.spread),
        "depth": depth,
    }
    return _validate_record(LOBSnapshotSchema, record, "LOBSnapshot")


def validate_feature_vector(feature: FeatureVector) -> bool:
    """Validate a FeatureVector instance and return True if it is valid."""
    record = {
        "ts": _to_timestamp(feature.ts),
        "instrument": feature.instrument,
        "returns": float(feature.returns),
        "volatility": float(feature.volatility),
        "z_score": float(feature.z_score),
        "spread": float(feature.spread),
        "imbalance": float(feature.imbalance),
    }
    if any(np.isnan(v) for v in record.values() if isinstance(v, float)):
        raise ValidationError("FeatureVector contains NaN values.")
    return _validate_record(FeatureVectorSchema, record, "FeatureVector")
