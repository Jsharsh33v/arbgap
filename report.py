"""Readable terminal output. Every number is also saved in results/*.csv."""

STRATEGIES = {
    "A": "fixed rule: buy 1 pm, sell 7 pm",
    "B": "plan with day-ahead prices",
    "C": "perfect hindsight",
}
HUB_NAMES = {"HB_HOUSTON": "Houston", "HB_NORTH": "North", "HB_SOUTH": "South", "HB_WEST": "West"}


def hub_name(node):
    return HUB_NAMES.get(node, node)


def usd(value, decimals=2):
    """-7.99 -> '-$7.99' instead of '$-7.99'."""
    text = f"${abs(value):,.{decimals}f}"
    return "-" + text if value < 0 else text


def p_text(p):
    """Very small p-values are easier to read as '< 0.001'."""
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def heading(title):
    print(f"\n{title}\n{'-' * len(title)}")


def print_setup(daily, eff, wear):
    heading("ArbGap backtest")
    print(f"   Data:     {daily['date'].nunique()} days x {daily['node'].nunique()} hubs "
          f"= {len(daily)} hub-days")
    print(f"   Battery:  {eff:.0%} efficiency, ${wear:g} wear per MWh sold, one trade a day")
    print(f"   Checks:   {3 * len(daily):,} plans replayed against the battery rules, "
          f"{daily['rule_breaks'].sum()} broken")


def print_profit_statement(statement, hub_days):
    heading("1. Profit statement ($ per MW per day, average over all hub-days)")
    print(f"   {'Strategy':<36}{'Revenue':>9}{'Energy':>9}{'Wear':>8}{'Profit':>9}   Days traded")
    for row in statement.itertuples():
        label = f"{row.strategy}  {STRATEGIES[row.strategy]}"
        print(f"   {label:<36}{row.revenue:>9.2f}{row.energy_cost:>9.2f}{row.wear_cost:>8.2f}"
              f"{row.profit:>9.2f}   {row.trade_days} of {hub_days}")


def print_gaps(summary):
    heading("2. Better decisions (B - A) vs. better information (C - B), $ per MW per day")
    print(f"   {'Hub':<10}{'Decisions':>11}{'Information':>13}{'B beats A':>13}{'C beats B':>12}")
    for node, group in summary.groupby("node", sort=False):
        gaps = group.set_index("comparison")
        decisions, information = gaps.loc["B_minus_A"], gaps.loc["C_minus_B"]
        print(f"   {hub_name(node):<10}{decisions['mean']:>11.2f}{information['mean']:>13.2f}"
              f"{decisions['pct_days_above_zero']:>12.0f}%{information['pct_days_above_zero']:>11.0f}%")
    print("   'B beats A' and 'C beats B' = share of days where that gap is above zero.")


def result_cells(row):
    significant = "yes" if row.p_value < 0.05 else "no"
    return f"{row.mean_difference:>8.2f}{row.t:>7.2f}{p_text(row.p_value):>10}{significant:>5}"


def print_tests(tests):
    heading("3. Hypothesis tests (one-sided paired t-tests)")
    print("   H1: B earns more than A.  H2: information (C - B) is worth more than decisions (B - A).")
    print(f"   {'':<10}{'---------- H1 ----------':>30}   {'---------- H2 ----------':>30}")
    print(f"   {'Hub':<10}{'gap':>8}{'t':>7}{'p':>10}{'sig':>5}   {'gap':>8}{'t':>7}{'p':>10}{'sig':>5}")
    for group, rows in tests.groupby("group", sort=False):
        h1, h2 = rows.iloc[0], rows.iloc[1]
        print(f"   {hub_name(group):<10}{result_cells(h1)}   {result_cells(h2)}")
    print(f"   n = {tests['n_days'].iloc[0]} days per test; sig = significant at the 5% level.")


def percent(value, base):
    """A percent change is misleading when the base is close to zero."""
    return "n/a" if base < 1 else f"{value:+.0f}%"


def print_business(business):
    mw, days = business["mw"].iloc[0], business["days"].iloc[0]
    heading(f"4. Business view: a {mw:g} MW battery over {days} days (modeled, May-July 2026)")
    print(f"   {'Hub':<10}{'B vs A':>9}{'C vs B':>9}{'Decisions worth':>18}{'Information worth':>20}")
    for row in business.itertuples():
        print(f"   {hub_name(row.group):<10}{percent(row.B_vs_A_pct, row.A):>9}"
              f"{percent(row.C_vs_B_pct, row.B):>9}{usd(row.decisions_total, 0):>18}"
              f"{usd(row.information_total, 0):>20}")
    print("   Information worth = the most a perfect forecast could have added (an upper bound).")
    print("   n/a = the base strategy earns under $1 per MW per day, so a % would mislead.")


def print_report(daily, summary, tests, statement, business, eff, wear):
    print_setup(daily, eff, wear)
    print_profit_statement(statement, len(daily))
    print_gaps(summary)
    print_tests(tests)
    print_business(business)
    print("\nModeled from historical ERCOT prices, not actual battery earnings.")
    print("Full tables in results/: summary.csv, ttests.csv, profit_statement.csv, business.csv")