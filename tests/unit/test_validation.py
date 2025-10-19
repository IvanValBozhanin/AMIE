from datetime import datetime

import numpy as np
import pytest

from amie.core.constants import Side
from amie.core.types import FeatureVector, LOBSnapshot, Tick
from amie.data.validation import (
    ValidationError,
    validate_feature_vector,
    validate_lob,
    validate_tick,
)


def _default_ts() -> datetime:
    return datetime(2024, 1, 1, 0, 0, 0)


def test_validate_tick_passes() -> None:
    tick = Tick(
        ts=_default_ts(),
        instrument="BTC-USD",
        price=50_000.0,
        qty=1.0,
        side=Side.BUY,
    )
    assert validate_tick(tick)


def test_validate_lob_passes() -> None:
    lob = LOBSnapshot(
        ts=_default_ts(),
        instrument="BTC-USD",
        bids=[(49_990.0, 1.0)],
        asks=[(50_010.0, 1.0)],
        spread=20.0,
    )
    assert validate_lob(lob)


def test_validate_tick_negative_price_raises_validation_error() -> None:
    tick = Tick(
        ts=_default_ts(),
        instrument="BTC-USD",
        price=-1.0,
        qty=1.0,
        side=Side.SELL,
    )
    with pytest.raises(ValidationError):
        validate_tick(tick)


def test_tick_schema_missing_field_raises_validation_error() -> None:
    tick = Tick(
        ts=_default_ts(),
        instrument="",
        price=50_000.0,
        qty=1.0,
        side=Side.BUY,
    )
    with pytest.raises(ValidationError):
        validate_tick(tick)


def test_feature_vector_null_values_raise_validation_error() -> None:
    feature = FeatureVector(
        ts=_default_ts(),
        instrument="BTC-USD",
        returns=np.nan,
        volatility=0.2,
        z_score=1.0,
        spread=20.0,
        imbalance=0.1,
    )
    with pytest.raises(ValidationError):
        validate_feature_vector(feature)
