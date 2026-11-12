from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from battery import Battery, check_plan
from trade import best_trade, realized


def run_day(da, rt, eff=0.85, wear=10):
      a = realized(13, 19, rt, eff, wear)  # Fixed hours
      buy, sell, expected = best_trade(da, eff, wear)
      b = realized(buy, sell, rt, eff, wear)  # Keep the day-ahead plan
      c_buy, c_sell, c = best_trade(rt, eff, wear)  # Hindsight
      assert c >= a and c >= b
      # Replay every plan against the battery rules (see battery.py).
      battery = Battery(eff=eff, wear=wear)
      rule_breaks = 0
      for plan in [(13, 19), (buy, sell), (c_buy, c_sell)]:
          trades = [] if plan == (None, None) else [plan]
          rule_breaks += len(check_plan(trades, battery, len(rt)))
      return {"A": a, "B": b, "C": c, "b_buy": buy, "b_sell": sell, "rule_breaks": rule_breaks}


def backtest(prices, eff=0.85, wear=10):
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
    print(f"Checked {3 * len(daily)} plans against the battery rules: "
        f"{daily['rule_breaks'].sum()} broken.")
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
    summary.to_csv("results/summary.csv", index=False)
    print(summary.round(2).to_string(index=False))
    means = by_date[["A", "B", "C"]].mean()
    print("Mean values ($ per MW per day):\n", means.round(2).to_string())
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
    print(tests.to_string(index=False, float_format="{:.4g}".format))
    return tests
