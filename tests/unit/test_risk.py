from __future__ import annotations

import pytest

from amie.strategy.risk import RiskManager


def test_position_capping() -> None:
    manager = RiskManager({"max_position_size": 5.0, "max_drawdown_pct": 0.5})

    proposed = 10.0
    adjusted = manager.check_position(proposed_position=proposed, current_equity=1_000.0)

    assert adjusted == pytest.approx(5.0)


def test_drawdown_guard_triggers_flat() -> None:
    manager = RiskManager({"max_position_size": 5.0, "max_drawdown_pct": 0.1})
    manager.update_drawdown(current_equity=80_000.0, peak_equity=100_000.0)

    assert manager.current_drawdown_pct == pytest.approx(0.2)
    adjusted = manager.check_position(proposed_position=2.0, current_equity=80_000.0)

    assert adjusted == pytest.approx(0.0)


def test_extreme_inputs_handled_gracefully() -> None:
    manager = RiskManager({"max_position_size": 1.5, "max_drawdown_pct": 0.5})
    manager.update_drawdown(current_equity=0.0, peak_equity=0.0)

    adjusted = manager.check_position(proposed_position=1_000_000.0, current_equity=0.0)

    assert adjusted == pytest.approx(1.5)
