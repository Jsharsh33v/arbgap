import pandas as pd
import pytest
from trade import best_trade, realized, realized_breakdown, trade_breakdown, trade_profit
from backtest import backtest, business_summary, paired_tests, run_day
from report import p_text, percent, usd
from sensitivity import sensitivity
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


def test_breakdown_adds_up():
    parts = trade_breakdown(20, 60)
    assert parts["revenue"] == pytest.approx(51)  # 0.85 MWh sold at $60
    assert parts["energy_cost"] == pytest.approx(20)
    assert parts["wear_cost"] == pytest.approx(8.5)  # $10 on 0.85 MWh
    assert parts["profit"] == pytest.approx(trade_profit(20, 60))
    no_trade = realized_breakdown(None, None, [30] * 24)
    assert all(value == 0 for value in no_trade.values())


def test_day_money_parts_match_profit():
    da = [30] * 24
    rt = [30] * 24
    da[1], da[20] = 1, 100
    rt[1], rt[20] = 100, 1
    day = run_day(da, rt)
    for name in ["A", "B", "C"]:
        money = day[f"{name}_revenue"] - day[f"{name}_energy_cost"] - day[f"{name}_wear_cost"]
        assert money == pytest.approx(day[name])


def test_business_summary_scales_with_battery_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # save results/business.csv in a temporary folder
    daily = pd.DataFrame({"date": ["2026-05-01", "2026-05-02"], "node": "HB_TEST",
                          "A": [10, 10], "B": [15, 15], "C": [30, 30]})
    one = business_summary(daily, mw=1).iloc[0]
    assert one["B_vs_A_pct"] == pytest.approx(50)
    assert one["C_vs_B_pct"] == pytest.approx(100)
    assert one["decisions_total"] == pytest.approx(10)  # $5 a day for 2 days
    two = business_summary(daily, mw=2).iloc[0]
    assert two["information_total"] == pytest.approx(2 * one["information_total"])


def fake_prices(days=3):
    """Made-up prices for a few days at one hub: rising during the day, with
    real-time prices a bit off from day-ahead in a different way each day."""
    rows = []
    for day in range(days):
        for hour in range(24):
            da = 20 + 3 * hour + day
            rt = da + (hour * 7 + day * 13) % 17 - 8
            rows.append({"date": f"2026-05-0{day + 1}", "node": "HB_TEST", "hour": hour,
                         "da_price": da, "rt_price": rt})
    return pd.DataFrame(rows)


def test_sensitivity_main_row_matches_backtest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # save results/sensitivity.csv in a temporary folder
    prices = fake_prices()
    table = sensitivity(prices, effs=[0.85], wears=[10])
    daily = backtest(prices, 0.85, 10, save=False)
    for name in ["A", "B", "C"]:
        assert table.loc[0, name] == pytest.approx(daily[name].mean())
    assert not (tmp_path / "results" / "daily.csv").exists()  # save=False wrote nothing


def test_report_formatting():
    assert usd(-7.99) == "-$7.99"
    assert usd(1032.43, 0) == "$1,032"
    assert p_text(0.0000002) == "< 0.001"
    assert p_text(0.05002) == "0.0500"
    assert percent(2570.7, 0.40) == "n/a"  # base under $1, so no percent
    assert percent(106.4, 6.24) == "+106%"
