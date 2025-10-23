from __future__ import annotations

import pytest

from amie.strategy.execution import ExecutionSimulator


def test_slippage_calculation_for_buy() -> None:
    simulator = ExecutionSimulator(
        {"slippage_bps": 5.0, "fee_bps": 0.0, "instrument": "TEST"}
    )
    fill = simulator.execute(order_qty=10.0, market_price=100.0, spread=0.2)

    expected_slippage = (5.0 / 10_000) * 100.0 + 0.2 / 2
    assert fill.price == pytest.approx(100.0 + expected_slippage)
    assert fill.slippage == pytest.approx(expected_slippage)


def test_fee_calculation() -> None:
    simulator = ExecutionSimulator(
        {"slippage_bps": 0.0, "fee_bps": 2.0, "instrument": "TEST"}
    )
    simulator.execute(order_qty=50.0, market_price=20.0, spread=0.0)

    expected_fee = (2.0 / 10_000) * (50.0 * 20.0)
    assert simulator.last_fee == pytest.approx(expected_fee)


def test_buy_sell_price_asymmetry() -> None:
    simulator = ExecutionSimulator(
        {"slippage_bps": 1.0, "fee_bps": 0.0, "instrument": "TEST"}
    )
    buy_fill = simulator.execute(order_qty=5.0, market_price=50.0, spread=0.1)
    sell_fill = simulator.execute(order_qty=-5.0, market_price=50.0, spread=0.1)

    assert buy_fill.price > 50.0
    assert sell_fill.price < 50.0
    assert buy_fill.slippage == pytest.approx(-sell_fill.slippage)


def test_higher_spread_increases_slippage() -> None:
    simulator = ExecutionSimulator(
        {"slippage_bps": 0.0, "fee_bps": 0.0, "instrument": "TEST"}
    )
    narrow_fill = simulator.execute(order_qty=1.0, market_price=100.0, spread=0.1)
    wide_fill = simulator.execute(order_qty=1.0, market_price=100.0, spread=0.5)

    assert wide_fill.slippage > narrow_fill.slippage
