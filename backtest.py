from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from battery import Battery, check_plan
from trade import best_trade, realized, realized_breakdown


def run_day(da, rt, eff=0.85, wear=10):
    a = realized(13, 19, rt, eff, wear)  # Fixed hours
    buy, sell, expected = best_trade(da, eff, wear)
    b = realized(buy, sell, rt, eff, wear)  # Keep the day-ahead plan
    c_buy, c_sell, c = best_trade(rt, eff, wear)  # Hindsight
    assert c >= a and c >= b
    plans = {"A": (13, 19), "B": (buy, sell), "C": (c_buy, c_sell)}
    # Replay every plan against the battery rules (see battery.py).
    battery = Battery(eff=eff, wear=wear)
    rule_breaks = 0
    for plan in plans.values():
        trades = [] if plan == (None, None) else [plan]
        rule_breaks += len(check_plan(trades, battery, len(rt)))
    row = {"A": a, "B": b, "C": c, "b_buy": buy, "b_sell": sell, "rule_breaks": rule_breaks}
    # Money in and out for each strategy, scored at real-time prices.
    for name, (buy_hour, sell_hour) in plans.items():
        parts = realized_breakdown(buy_hour, sell_hour, rt, eff, wear)
        for part in ["revenue", "energy_cost", "wear_cost"]:
            row[f"{name}_{part}"] = parts[part]
        row[f"{name}_traded"] = buy_hour is not None
    return row


def backtest(prices, eff=0.85, wear=10, save=True):
    rows = []
    for (date, node), day in prices.groupby(["date", "node"]):
        day = day.sort_values("hour")
        if day["hour"].tolist() != list(range(24)):
            raise ValueError(f"{date} / {node} needs all 24 hours, with no duplicates.")
        da = day["da_price"].tolist()
        rt = day["rt_price"].tolist()
        row = {"date": date, "node": node}
        row.update(run_day(da, rt, eff, wear))
        rows.append(row)
    daily = pd.DataFrame(rows)
    # Whole-hour numbers, with blanks when B skips a trade.
    for column in ["b_buy", "b_sell"]:
        daily[column] = daily[column].astype("Int64")
    daily["B_minus_A"] = daily["B"] - daily["A"]
    daily["C_minus_B"] = daily["C"] - daily["B"]
    if save:  # sensitivity.py runs many backtests and keeps daily.csv unchanged
        Path("results").mkdir(exist_ok=True)
        daily.to_csv("results/daily.csv", index=False)
    return daily


def summarize(daily):
    columns = ["A", "B", "C", "B_minus_A", "C_minus_B"]
    by_date = daily.groupby("date")[columns].mean()
    groups = [("All hubs", by_date)]
    for node, group in daily.groupby("node"):
        groups.append((node, group))
    rows = []
    for node, group in groups:
        for gap in ["B_minus_A", "C_minus_B"]:
            values = group[gap]
            rows.append({"node": node, "comparison": gap,
                         "mean": values.mean(), "median": values.median(),
                         "pct_days_above_zero": (values > 0).mean() * 100})
    summary = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/summary.csv", index=False)
    means = by_date[["A", "B", "C"]].mean()
    ax = means.plot.bar(rot=0)
    ax.set_ylabel("$ per MW per day")
    ax.set_title("A: fixed hours | B: day-ahead plan | C: hindsight")
    plt.tight_layout()
    plt.savefig("results/mean_values.png")
    plt.close()
    return summary


def paired_tests(daily):
    """One-sided paired t-tests for H1 and H2, overall and for each hub.

    H1: B earns more than A.  H2: C - B (information) is bigger than B - A (decisions).
    "All hubs" uses the 92 daily averages (four hubs averaged per date first).
    """
    groups = [("All hubs", daily.groupby("date")[["A", "B", "C"]].mean())]
    for node, group in daily.groupby("node"):
        groups.append((node, group))
    rows = []
    for name, data in groups:
        decisions = data["B"] - data["A"]  # value of better decisions
        information = data["C"] - data["B"]  # value of information
        pairs = [("H1: B > A", data["B"], data["A"]),
                 ("H2: C-B > B-A", information, decisions)]
        for hypothesis, first, second in pairs:
            result = stats.ttest_rel(first, second, alternative="greater")
            rows.append({"group": name, "hypothesis": hypothesis,
                         "mean_difference": (first - second).mean(),
                         "t": result.statistic, "p_value": result.pvalue,
                         "n_days": len(first)})
    tests = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    tests.to_csv("results/ttests.csv", index=False)
    return tests


def profit_statement(daily):
    """Average revenue, costs and profit per MW per day for A, B and C (all hub-days)."""
    rows = []
    for name in ["A", "B", "C"]:
        rows.append({"strategy": name,
                     "revenue": daily[f"{name}_revenue"].mean(),
                     "energy_cost": daily[f"{name}_energy_cost"].mean(),
                     "wear_cost": daily[f"{name}_wear_cost"].mean(),
                     "profit": daily[name].mean(),
                     "trade_days": daily[f"{name}_traded"].sum()})
    statement = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    statement.to_csv("results/profit_statement.csv", index=False)
    return statement


def business_summary(daily, mw=1.0):
    """Percent gains and money totals, overall and for each hub.

    Totals scale the per-MW results to a battery of `mw` MW over the days in the data.
    They are modeled estimates for May-July 2026, not a full year.
    """
    groups = [("All hubs", daily.groupby("date")[["A", "B", "C"]].mean())]
    for node, group in daily.groupby("node"):
        groups.append((node, group))
    rows = []
    for name, data in groups:
        a, b, c = data["A"].mean(), data["B"].mean(), data["C"].mean()
        days = len(data)
        rows.append({"group": name, "A": a, "B": b, "C": c,
                     # Percent gains only make sense when the base is above zero.
                     "B_vs_A_pct": (b - a) / a * 100 if a > 0 else float("nan"),
                     "C_vs_B_pct": (c - b) / b * 100 if b > 0 else float("nan"),
                     "days": days, "mw": mw,
                     "decisions_total": (b - a) * mw * days,
                     "information_total": (c - b) * mw * days})
    summary = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/business.csv", index=False)
    return summary
