"""A/B/C comparison for a bigger battery that can trade several times a day.

    python multi_backtest.py --capacity 2 --trades 2

A buys in the hours starting 1 pm and sells in the hours starting 7 pm (one pair per
trade); B plans with day-ahead prices; C plans with real-time prices (hindsight).
B and C use the exact dynamic-programming solver. All three are paid real-time prices.
"""
import argparse
from pathlib import Path
import pandas as pd
from scipy import stats
from battery import Battery, check_plan
from multi import best_plan_dp, plan_profit, units
from report import heading, hub_name, p_text


def fixed_plan(battery):
    """Rule A for any battery: buy 1 pm, 2 pm, ... and sell 7 pm, 8 pm, ..."""
    count = min(battery.max_trades, units(battery), 5)
    return [(13 + i, 19 + i) for i in range(count)]


def multi_backtest(prices, battery):
    rows = []
    for (date, node), day in prices.groupby(["date", "node"]):
        day = day.sort_values("hour")
        da, rt = day["da_price"].tolist(), day["rt_price"].tolist()
        plans = {"A": fixed_plan(battery), "B": best_plan_dp(da, battery)[0],
                 "C": best_plan_dp(rt, battery)[0]}
        row = {"date": date, "node": node}
        for name, plan in plans.items():
            row[name] = plan_profit(plan, rt, battery)
            row[f"{name}_trades"] = len(plan)
        row["rule_breaks"] = sum(len(check_plan(p, battery, len(rt))) for p in plans.values())
        rows.append(row)
    return pd.DataFrame(rows)


def multi_tests(daily):
    """H1 (B > A) and H2 (C - B > B - A), overall on daily averages and per hub."""
    groups = [("All hubs", daily.groupby("date")[["A", "B", "C"]].mean())]
    groups += list(daily.groupby("node"))
    rows = []
    for name, data in groups:
        decisions, information = data["B"] - data["A"], data["C"] - data["B"]
        h1 = stats.ttest_rel(data["B"], data["A"], alternative="greater")
        h2 = stats.ttest_rel(information, decisions, alternative="greater")
        rows.append({"group": name, "A": data["A"].mean(), "B": data["B"].mean(),
                     "C": data["C"].mean(), "decisions": decisions.mean(),
                     "information": information.mean(), "H1_p": h1.pvalue, "H2_p": h2.pvalue})
    return pd.DataFrame(rows)


def print_multi(daily, tests, battery):
    trades = "trade" if battery.max_trades == 1 else "trades"
    heading(f"Battery: {battery.capacity_mwh:g} MWh, up to {battery.max_trades} {trades} a day "
            f"($ per MW per day)")
    print(f"   Checks: {3 * len(daily):,} plans replayed, {daily['rule_breaks'].sum()} broken. "
          f"B trades {daily['B_trades'].mean():.2f} times a day on average.")
    print(f"   {'Hub':<10}{'A':>8}{'B':>8}{'C':>8}{'Decisions':>11}{'Information':>13}"
          f"{'H1 p':>10}{'H2 p':>10}")
    for row in tests.itertuples():
        print(f"   {hub_name(row.group):<10}{row.A:>8.2f}{row.B:>8.2f}{row.C:>8.2f}"
              f"{row.decisions:>11.2f}{row.information:>13.2f}{p_text(row.H1_p):>10}"
              f"{p_text(row.H2_p):>10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=float, default=2.0, help="MWh the battery holds")
    parser.add_argument("--trades", type=int, default=2, help="most trades per day")
    args = parser.parse_args()
    if not args.capacity > 0 or args.trades < 1:
        parser.error("Use a capacity above 0 and at least 1 trade per day.")
    battery = Battery(capacity_mwh=args.capacity, max_trades=args.trades)
    daily = multi_backtest(pd.read_csv("data/prices.csv"), battery)
    tests = multi_tests(daily)
    Path("results").mkdir(exist_ok=True)
    daily.to_csv("results/multi_daily.csv", index=False)
    tests.to_csv("results/multi_ttests.csv", index=False)
    print_multi(daily, tests, battery)