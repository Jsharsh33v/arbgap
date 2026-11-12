"""Rerun the backtest with different efficiency and wear-cost assumptions.

Run after the main backtest:  python sensitivity.py
"""
from pathlib import Path
import pandas as pd
from scipy import stats
from backtest import backtest
from report import heading, p_text

EFFS = [0.80, 0.85, 0.90, 0.95]  # share of bought energy that can be sold
WEARS = [0, 10, 20]  # wear cost in $ per MWh sold


def sensitivity(prices, effs=EFFS, wears=WEARS):
    """One row per (efficiency, wear) pair: mean values and the H1 and H2 tests."""
    rows = []
    for eff in effs:
        for wear in wears:
            daily = backtest(prices, eff, wear, save=False)
            by_date = daily.groupby("date")[["A", "B", "C"]].mean()
            decisions = by_date["B"] - by_date["A"]
            information = by_date["C"] - by_date["B"]
            h1 = stats.ttest_rel(by_date["B"], by_date["A"], alternative="greater")
            h2 = stats.ttest_rel(information, decisions, alternative="greater")
            rows.append({"eff": eff, "wear": wear,
                         "A": by_date["A"].mean(), "B": by_date["B"].mean(),
                         "C": by_date["C"].mean(),
                         "B_minus_A": decisions.mean(), "C_minus_B": information.mean(),
                         "H1_p": h1.pvalue, "H2_p": h2.pvalue,
                         "H2_holds_in_means": information.mean() > decisions.mean(),
                         "rule_breaks": daily["rule_breaks"].sum(),
                         "plans_checked": 3 * len(daily)})
    table = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/sensitivity.csv", index=False)
    return table


def print_sensitivity(table):
    heading("Sensitivity check: does information still beat decisions? ($ per MW per day)")
    print(f"   {'Efficiency':>10}{'Wear':>7}{'Decisions':>11}{'Information':>13}"
          f"{'Info > decisions?':>19}{'H2 p':>9}{'sig':>5}")
    for row in table.itertuples():
        holds = "yes" if row.H2_holds_in_means else "no"
        significant = "yes" if row.H2_p < 0.05 else "no"
        print(f"   {row.eff:>10.0%}{'$' + format(row.wear, 'g'):>7}{row.B_minus_A:>11.2f}"
              f"{row.C_minus_B:>13.2f}{holds:>19}{p_text(row.H2_p):>9}{significant:>5}")
    print(f"   H1 (B beats A) is significant in {(table['H1_p'] < 0.05).sum()} of {len(table)} runs.")
    print(f"   Checks: {table['plans_checked'].sum():,} plans replayed, "
          f"{table['rule_breaks'].sum()} broken. Full table: results/sensitivity.csv")


if __name__ == "__main__":
    print_sensitivity(sensitivity(pd.read_csv("data/prices.csv")))