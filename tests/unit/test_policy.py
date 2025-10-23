from __future__ import annotations

from datetime import datetime

import pytest

from amie.core.types import Signal
from amie.strategy.policy import SignalPolicy


def _make_signal(score: float, uncertainty: float) -> Signal:
    return Signal(
        ts=datetime(2024, 1, 1),
        instrument="TEST",
        score=score,
        uncertainty=uncertainty,
        model_version="v1",
    )


def test_long_signal_above_threshold() -> None:
    policy = SignalPolicy({"threshold_multiplier": 2.0, "max_position_size": 1.0})
    signal = _make_signal(score=3.0, uncertainty=1.0)

    assert policy.compute_position(signal) == pytest.approx(1.0)


def test_flat_signal_within_threshold_band() -> None:
    policy = SignalPolicy({"threshold_multiplier": 2.0, "max_position_size": 1.0})
    signal = _make_signal(score=0.5, uncertainty=1.0)

    assert policy.compute_position(signal) == pytest.approx(0.0)


def test_position_capped_at_maximum() -> None:
    policy = SignalPolicy({"threshold_multiplier": 2.0, "max_position_size": 1.5})
    signal = _make_signal(score=10.0, uncertainty=0.1)

    assert policy.compute_position(signal) == pytest.approx(1.5)


def test_higher_uncertainty_reduces_position() -> None:
    policy = SignalPolicy({"threshold_multiplier": 2.0, "max_position_size": 10.0})
    low_uncertainty_signal = _make_signal(score=5.0, uncertainty=0.5)
    high_uncertainty_signal = _make_signal(score=5.0, uncertainty=1.0)

    low_uncertainty_position = policy.compute_position(low_uncertainty_signal)
    high_uncertainty_position = policy.compute_position(high_uncertainty_signal)

    assert abs(low_uncertainty_position) > abs(high_uncertainty_position)
