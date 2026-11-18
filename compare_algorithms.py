"""Compare the solvers on the real price data: how long each takes and how close it gets.

    python compare_algorithms.py          # DP, CP-SAT, time-limited CP-SAT, hill climbing
    python compare_algorithms.py --z3     # also Z3 (pip install z3-solver), slow

Part 1 solves every price-day (368 hub-days x day-ahead and real-time = 736 days).
Part 2 plans one long stretch of Houston real-time prices at once (1 to 92 days), which
is where the exact and fast methods start to differ. Times are on one CPU core.
"""
import argparse
import time
from pathlib import Path
import pandas as pd
from battery import Battery, check_plan
from multi import SCALE, best_plan_cpsat, best_plan_dp, hill_climb, plan_profit, to_trades, units

SIZES = [(1, 1), (2, 2), (4, 4)]  # (MWh, most trades per day)
HORIZONS = [1, 3, 7, 14, 30, 92]  # days planned at once in part 2
TIME_LIMITS = [0.01, 0.1]  # seconds for the time-limited CP-SAT runs


def timed(solve, *args, **kwargs):
    """Run a solver and return (its answer, milliseconds it took)."""
    start = time.perf_counter()
    answer = solve(*args, **kwargs)
    return answer, 1000 * (time.perf_counter() - start)


def price_days(prices):
    """Every day of day-ahead and every day of real-time prices, as 24-hour lists."""
    days = []
    for _, day in prices.groupby(["date", "node"]):
        day = day.sort_values("hour")
        days += [day["da_price"].tolist(), day["rt_price"].tolist()]
    return days


def daily_comparison(prices):
    rows = []
    days = price_days(prices)
    for mwh, trades in SIZES:
        battery = Battery(capacity_mwh=mwh, max_trades=trades)
        for day in days:
            (dp_plan, best), dp_ms = timed(best_plan_dp, day, battery)
            (_, cp_profit, status), cp_ms = timed(best_plan_cpsat, day, battery)
            (hill_plan, hill_profit), hill_ms = timed(hill_climb, day, battery)
            rows.append({"capacity_mwh": mwh, "max_trades": trades, "best": best,
                         "cpsat_status": status, "cpsat_gap": best - cp_profit,
                         "hill_gap": best - hill_profit, "dp_ms": dp_ms, "cpsat_ms": cp_ms,
                         "hill_ms": hill_ms, "rule_breaks": len(check_plan(dp_plan, battery))
                         + len(check_plan(hill_plan, battery))})
    return pd.DataFrame(rows)


def best_plan_z3(prices, battery, timeout_s=30):
    """The same model as best_plan_cpsat, written for the Z3 SMT solver."""
    import z3
    hours, most = len(prices), units(battery)
    opt = z3.Optimize()
    opt.set("timeout", int(timeout_s * 1000))
    buy = [z3.Bool(f"buy_{t}") for t in range(hours)]
    sell = [z3.Bool(f"sell_{t}") for t in range(hours)]
    level = [z3.Int(f"level_{t}") for t in range(hours)]
    for t in range(hours):
        opt.add(z3.Not(z3.And(buy[t], sell[t])))
        before = level[t - 1] if t > 0 else 0
        opt.add(level[t] == before + z3.If(buy[t], 1, 0) - z3.If(sell[t], 1, 0))
        opt.add(level[t] >= 0, level[t] <= most)
    opt.add(level[-1] == 0)
    opt.add(z3.Sum([z3.If(b, 1, 0) for b in buy]) <= battery.max_trades)
    money_buy = [round((-p - battery.trade_cost) * SCALE) for p in prices]
    money_sell = [round(battery.eff * (p - battery.wear) * SCALE) for p in prices]
    opt.maximize(z3.Sum([z3.If(buy[t], money_buy[t], 0) + z3.If(sell[t], money_sell[t], 0)
                         for t in range(hours)]))
    if opt.check() != z3.sat:
        return [], 0.0, "no answer"
    model = opt.model()
    actions = [1 if z3.is_true(model.eval(buy[t])) else
               -1 if z3.is_true(model.eval(sell[t])) else 0 for t in range(hours)]
    trades = to_trades(actions)
    return trades, plan_profit(trades, prices, battery), "done"


def horizon_comparison(prices, use_z3):
    """Plan several days at once (2 MWh, 2 trades a day on average, energy can stay
    in the battery overnight), Houston real-time prices."""
    houston = prices[prices["node"] == "HB_HOUSTON"].sort_values(["date", "hour"])
    rt = houston["rt_price"].tolist()
    rows = []
    for days in HORIZONS:
        stretch = rt[:24 * days]
        battery = Battery(capacity_mwh=2, max_trades=2 * days)
        (_, best), dp_ms = timed(best_plan_dp, stretch, battery)
        row = {"days": days, "hours": len(stretch), "best": best, "dp_ms": dp_ms}
        (_, profit, status), row["cpsat_ms"] = timed(best_plan_cpsat, stretch, battery)
        row["cpsat_status"], row["cpsat_gap"] = status, best - profit
        for limit in TIME_LIMITS:
            _, profit, status = best_plan_cpsat(stretch, battery, time_limit=limit)
            row[f"cpsat_{limit}s_status"] = status
            row[f"cpsat_{limit}s_gap"] = best - profit if status != "UNKNOWN" else None
        if days <= 14:  # hill climbing gets slow on long stretches
            (_, profit), row["hill_ms"] = timed(hill_climb, stretch, battery)
            row["hill_gap"] = best - profit
        if use_z3 and days <= 7:
            (_, profit, status), row["z3_ms"] = timed(best_plan_z3, stretch, battery)
            row["z3_status"] = status
            row["z3_gap"] = best - profit if status == "done" else None
        rows.append(row)
    return pd.DataFrame(rows)


def print_daily(daily):
    print("\nPart 1: one day at a time (736 price-days), times in milliseconds per day")
    print(f"   {'Battery':<18}{'DP':>8}{'CP-SAT':>9}{'Hill':>8}   Hill found the best plan")
    for (mwh, trades), group in daily.groupby(["capacity_mwh", "max_trades"]):
        exact = (group["hill_gap"] < 0.005).sum()
        label = f"{mwh} MWh, {trades} trade" + ("s" if trades > 1 else "")
        print(f"   {label:<18}{group['dp_ms'].mean():>8.2f}"
              f"{group['cpsat_ms'].mean():>9.2f}{group['hill_ms'].mean():>8.2f}   "
              f"{exact} of {len(group)} days (worst ${group['hill_gap'].max():.2f} short)")
    print(f"   CP-SAT proved its plan best on {(daily['cpsat_status'] == 'OPTIMAL').sum()} of "
          f"{len(daily)} runs; it matched DP on {(daily['cpsat_gap'].abs() < 0.005).sum()}. "
          f"Broken plans: {daily['rule_breaks'].sum()}.")


def short(gap):
    """How far a plan is below the best one, in words."""
    if pd.isna(gap):
        return "no plan"
    return "best" if gap < 0.005 else f"${gap:.2f} short"


def print_horizons(table):
    print("\nPart 2: planning many days at once (Houston, 2 MWh, 2 trades a day)")
    print("   'no plan' = stopped before finding any plan; '-' = not run (too slow)")
    print(f"   {'Days':>4}{'Best plan':>11}{'DP':>9}{'CP-SAT':>9}"
          + "".join(f"{f'CP-SAT {t}s':>16}" for t in TIME_LIMITS)
          + f"{'Hill climbing':>24}" + (f"{'Z3':>22}" if "z3_ms" in table else ""))
    for _, row in table.iterrows():
        line = (f"   {row['days']:>4}{row['best']:>11.2f}{row['dp_ms']:>7.0f}ms"
                f"{row['cpsat_ms']:>7.0f}ms")
        line += "".join(f"{short(row[f'cpsat_{t}s_gap']):>16}" for t in TIME_LIMITS)
        hill = "-" if pd.isna(row.get("hill_ms")) else (
            f"{short(row['hill_gap'])}, {row['hill_ms']:.0f}ms")
        line += f"{hill:>24}"
        if "z3_ms" in table:
            z3 = "-" if pd.isna(row["z3_ms"]) else f"{short(row['z3_gap'])}, {row['z3_ms']:.0f}ms"
            line += f"{z3:>22}"
        print(line)
    print(f"   CP-SAT with no limit proved the best plan for every stretch: "
          f"{(table['cpsat_status'] == 'OPTIMAL').all()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--z3", action="store_true", help="also time Z3 (needs z3-solver)")
    args = parser.parse_args()
    prices = pd.read_csv("data/prices.csv")
    Path("results").mkdir(exist_ok=True)
    daily = daily_comparison(prices)
    daily.to_csv("results/algorithms_daily.csv", index=False)
    print_daily(daily)
    horizons = horizon_comparison(prices, args.z3)
    horizons.to_csv("results/algorithms_horizons.csv", index=False)
    print_horizons(horizons)