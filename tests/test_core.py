import pytest
from trade import best_trade, realized, trade_profit
from backtest import run_day


def test_five_hour_example():
    buy, sell, profit = best_trade([30, 20, 50, 10, 40])
    assert (buy, sell) == (3, 4)
    assert profit == pytest.approx(15.5)


def test_flat_prices():
    assert best_trade([30] * 24) == (None, None, 0.0)
    assert realized(None, None, [30] * 24) == 0


def test_realized_loss():
    assert realized(3, 4, [30, 20, 50, 25, 35]) == pytest.approx(-3.75)


def test_break_even():
    assert trade_profit(20, 20 / 0.85 + 10) == pytest.approx(0)


def test_hindsight_on_two_fake_days():
    da = [30] * 24
    rt = [30] * 24
    first = run_day(da, rt)
    da[1], da[20] = 1, 100
    rt[1], rt[20] = 100, 1
    second = run_day(da, rt)
    for day in [first, second]:
        assert day["C"] >= day["A"]
        assert day["C"] >= day["B"]
    assert first["A"] == pytest.approx(-13)
    assert first["B"] == 0
    assert (first["b_buy"], first["b_sell"]) == (None, None)
    assert (second["b_buy"], second["b_sell"]) == (1, 20)
    assert second["B"] == pytest.approx(-107.65)
