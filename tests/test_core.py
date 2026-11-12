import pandas as pd
import pytest
from trade import best_trade, realized, trade_profit
from backtest import paired_tests, run_day
from battery import Battery, check_plan


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
    assert first["rule_breaks"] == 0 and second["rule_breaks"] == 0


def test_check_plan_accepts_valid_plans():
      assert check_plan([(3, 4)], Battery()) == []
      assert check_plan([], Battery()) == []  # skipping the day is always allowed
      assert check_plan([(1, 2), (3, 4)], Battery(max_trades=2)) == []


def test_check_plan_catches_broken_rules():
      two_trades = Battery(max_trades=2)
      # Cheating: buy at 1 and 2 before selling = 1.7 MWh in a 1 MWh battery.
      problems = check_plan([(1, 5), (2, 6)], two_trades)
      assert any("over capacity" in problem for problem in problems)
      assert check_plan([(5, 3)], Battery())  # sells before it buys
      assert check_plan([(3, 30)], Battery())  # hour 30 is not in the day
      assert check_plan([(1, 2), (3, 4)], Battery())  # 2 trades, only 1 allowed
      assert check_plan([(1, 3), (3, 4)], two_trades)  # buys and sells in hour 3
      assert check_plan([(1, 2)], Battery(capacity_mwh=0.5))  # battery too small


def test_paired_tests_find_clear_gain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # save results/ttests.csv in a temporary folder
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    daily = pd.DataFrame({
        "date": [f"2026-05-0{i + 1}" for i in range(8)],
        "node": "HB_TEST",
        "A": a,
        "B": [x + 2 + 0.1 * i for i, x in enumerate(a)],  # B always beats A
        "C": [x + 6 + 0.35 * i for i, x in enumerate(a)],  # C - B always beats B - A
    })
    tests = paired_tests(daily)
    assert (tests["p_value"] < 0.05).all()
    assert (tests["n_days"] == 8).all()
