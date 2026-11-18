import random
import pandas as pd
import pytest
from battery import Battery, check_plan
from multi import (best_plan_brute, best_plan_cpsat, best_plan_dp, hill_climb, plan_profit,
                   to_trades, units)
from multi_backtest import fixed_plan, multi_backtest, multi_tests
from trade import best_trade


def random_days(count, hours, seed=1):
    """Made-up price days with spikes and dips, the same every run."""
    rng = random.Random(seed)
    return [[rng.uniform(-20, 150) for _ in range(hours)] for _ in range(count)]


def test_units():
    assert units(Battery(capacity_mwh=1)) == 1  # 0.85 MWh fits, 1.7 does not
    assert units(Battery(capacity_mwh=2)) == 2
    assert units(Battery(capacity_mwh=0.85)) == 1


def test_to_trades_pairs_oldest_energy_first():
    assert to_trades([1, 1, 0, -1, -1]) == [(0, 3), (1, 4)]
    assert to_trades([0, 0, 0]) == []


def test_exact_solvers_agree_with_brute_force():
    """Brute force tries every plan, so it is the answer key on small days."""
    batteries = [Battery(), Battery(capacity_mwh=2, max_trades=2),
                 Battery(capacity_mwh=2, max_trades=3, trade_cost=5)]
    for battery in batteries:
        for prices in random_days(30, 8):
            _, brute = best_plan_brute(prices, battery)
            dp_plan, dp = best_plan_dp(prices, battery)
            cp_plan, cp, status = best_plan_cpsat(prices, battery)
            assert status == "OPTIMAL"
            assert dp == pytest.approx(brute, abs=0.01)
            assert cp == pytest.approx(brute, abs=0.01)  # CP-SAT rounds to 1/10 cent
            assert check_plan(dp_plan, battery, len(prices)) == []
            assert check_plan(cp_plan, battery, len(prices)) == []


def test_one_trade_matches_original_model():
    """With the default battery, the new solvers give the old best_trade answer."""
    for prices in random_days(50, 24, seed=2):
        _, _, old = best_trade(prices)
        assert best_plan_dp(prices, Battery())[1] == pytest.approx(old)


def test_hill_climbing_plans_are_valid_and_never_beat_the_best():
    battery = Battery(capacity_mwh=4, max_trades=4)
    for prices in random_days(30, 24, seed=3):
        plan, profit = hill_climb(prices, battery)
        assert check_plan(plan, battery, len(prices)) == []
        assert profit == pytest.approx(plan_profit(plan, prices, battery))
        assert profit <= best_plan_dp(prices, battery)[1] + 1e-6


def test_no_trade_when_nothing_pays():
    for solve in [best_plan_dp, hill_climb]:
        assert solve([30.0] * 24, Battery(capacity_mwh=2, max_trades=2)) == ([], 0.0)


def test_bigger_battery_backtest():
    battery = Battery(capacity_mwh=2, max_trades=2)
    assert fixed_plan(battery) == [(13, 19), (14, 20)]
    rows = []
    for day, prices in enumerate(random_days(3, 24, seed=4)):
        for hour, price in enumerate(prices):
            rows.append({"date": f"2026-05-0{day + 1}", "node": "HB_TEST", "hour": hour,
                         "da_price": price, "rt_price": price + (hour * 7 + day) % 17 - 8})
    daily = multi_backtest(pd.DataFrame(rows), battery)
    assert daily["rule_breaks"].sum() == 0
    assert (daily["C"] >= daily["B"] - 1e-9).all()  # hindsight is never worse
    assert len(multi_tests(daily)) == 2  # all hubs + the one fake hub