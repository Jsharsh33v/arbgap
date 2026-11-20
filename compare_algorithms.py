"""Compare the solvers on the real price data: how long each takes and how close it gets.

    python compare_algorithms.py                    # every day, then long stretches
    python compare_algorithms.py --z3               # also Z3 (pip install z3-solver), slow
    python compare_algorithms.py --date 2026-07-01 --node HB_HOUSTON
    python compare_algorithms.py --date 2026-07-01 --node HB_HOUSTON --neighbors add

With --date and --node it solves that one day with every method and prints each plan.
Without them, part 1 solves every price-day (368 hub-days x day-ahead and real-time =
736 days) and part 2 plans one long stretch of Houston real-time prices at once (1 to
92 days), which is where the methods start to differ. Times are on one CPU core.
"""
import argparse
import time
from pathlib import Path
import pandas as pd
from battery import Battery, check_plan
from multi import (NEIGHBORS, SCALE, best_plan_brute, best_plan_cpsat, best_plan_dp, hill_climb,
                   plan_profit, to_trades, units)
from prepare_data import HUBS
from report import heading, hub_name

SIZES = [(1, 1), (2, 2), (4, 4), (4, 8)]  # (MWh, most trades per day)
HORIZONS = [1, 3, 7, 14, 30, 92]  # days planned at once in part 2
TIME_LIMITS = [0.01, 0.1]  # seconds for the time-limited CP-SAT runs
SWAP_DAYS = 14  # "swap" hill climbing gets slow on longer stretches
BRUTE_LIMIT = 2_000_000  # brute force only runs when the day has at most this many plans


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


def count_plans(hours, battery):
    """How many valid plans a day has: the number brute force has to try."""
    most, limit = units(battery), battery.max_trades
    ways = {(0, 0): 1}  # (energy stored, trades used) -> number of ways to get there
    for _ in range(hours):
        new = {}
        for (level, used), count in ways.items():
            states = [(level, used)]  # wait
            if level < most and used < limit:
                states.append((level + 1, used + 1))  # buy
            if level > 0:
                states.append((level - 1, used))  # sell
            for state in states:
                new[state] = new.get(state, 0) + count
        ways = new
    return sum(count for (level, _), count in ways.items() if level == 0)


def daily_comparison(prices):
    rows = []
    days = price_days(prices)
    for mwh, trades in SIZES:
        battery = Battery(capacity_mwh=mwh, max_trades=trades)
        for day in days:
            (dp_plan, best), dp_ms = timed(best_plan_dp, day, battery)
            (_, cp_profit, status), cp_ms = timed(best_plan_cpsat, day, battery)
            row = {"capacity_mwh": mwh, "max_trades": trades, "best": best,
                   "cpsat_status": status, "cpsat_gap": best - cp_profit, "dp_ms": dp_ms,
                   "cpsat_ms": cp_ms, "rule_breaks": len(check_plan(dp_plan, battery))}
            for name in NEIGHBORS:
                (plan, profit), row[f"hill_{name}_ms"] = timed(hill_climb, day, battery, name)
                row[f"hill_{name}_gap"] = best - profit
                row["rule_breaks"] += len(check_plan(plan, battery))
            rows.append(row)
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
        for name in NEIGHBORS:
            if name == "swap" and days > SWAP_DAYS:
                continue
            (plan, profit), row[f"hill_{name}_ms"] = timed(hill_climb, stretch, battery, name)
            row[f"hill_{name}_gap"] = best - profit
            row[f"hill_{name}_breaks"] = len(check_plan(plan, battery, len(stretch)))
        if use_z3 and days <= 7:
            (_, profit, status), row["z3_ms"] = timed(best_plan_z3, stretch, battery)
            row["z3_status"] = status
            row["z3_gap"] = best - profit if status == "done" else None
        rows.append(row)
    return pd.DataFrame(rows)


def quality(gap, best):
    """How good a plan is: 'best', or its profit as a percent of the best plan's profit."""
    if pd.isna(gap):
        return "no plan"
    if gap < 0.005:
        return "best"
    return f"{100 * (best - gap) / best:.1f}%" if best > 0 else "below best"


def print_daily(daily):
    days = len(daily) // len(SIZES)
    print(f"\nPart 1: one day at a time ({days} price-days), times in milliseconds per day")
    print("   Hill climbing columns: days it found the best plan (worst miss), time")
    print(f"   {'Battery':<17}{'DP':>6}{'CP-SAT':>8}"
          + "".join(f"   {name:<25}" for name in NEIGHBORS).rstrip())
    for (mwh, trades), group in daily.groupby(["capacity_mwh", "max_trades"]):
        label = f"{mwh} MWh, {trades} trade" + ("s" if trades > 1 else "")
        line = f"   {label:<17}{group['dp_ms'].mean():>6.2f}{group['cpsat_ms'].mean():>8.2f}"
        for name in NEIGHBORS:
            gaps = group[f"hill_{name}_gap"]
            miss = "none" if gaps.max() < 0.005 else f"${gaps.max():.2f}"
            cell = f"{(gaps < 0.005).sum()} ({miss}) {group[f'hill_{name}_ms'].mean():.2f}ms"
            line += f"   {cell:<25}"
        print(line.rstrip())
    print(f"   CP-SAT proved its plan best on {(daily['cpsat_status'] == 'OPTIMAL').sum()} of "
          f"{len(daily)} runs; it matched DP on {(daily['cpsat_gap'].abs() < 0.005).sum()}. "
          f"Broken plans: {daily['rule_breaks'].sum()}.")


def print_horizons(table):
    print("\nPart 2: planning many days at once (Houston, 2 MWh, 2 trades a day)")
    print("   'best' = same profit as the best plan; a percent = share of the best profit;")
    print("   'no plan' = stopped before finding any plan; '-' = not run (too slow)")
    print(f"   {'Days':>4}{'Best plan':>11}{'DP':>9}{'CP-SAT':>9}"
          + "".join(f"{f'CP-SAT {t}s':>13}" for t in TIME_LIMITS)
          + "".join(f"{f'Hill {name}':>17}" for name in NEIGHBORS)
          + (f"{'Z3':>17}" if "z3_ms" in table else ""))
    for _, row in table.iterrows():
        line = (f"   {row['days']:>4}{row['best']:>11.2f}{row['dp_ms']:>7.0f}ms"
                f"{row['cpsat_ms']:>7.0f}ms")
        line += "".join(f"{quality(row[f'cpsat_{t}s_gap'], row['best']):>13}"
                        for t in TIME_LIMITS)
        names = [f"hill_{name}" for name in NEIGHBORS] + (["z3"] if "z3_ms" in table else [])
        for name in names:
            cell = "-" if pd.isna(row[f"{name}_ms"]) else (
                f"{quality(row[f'{name}_gap'], row['best'])}, {row[f'{name}_ms']:.0f}ms")
            line += f"{cell:>17}"
        print(line)
    breaks = int(sum(table[f"hill_{name}_breaks"].sum() for name in NEIGHBORS))
    print(f"   CP-SAT with no limit proved the best plan for every stretch: "
          f"{(table['cpsat_status'] == 'OPTIMAL').all()}. Broken hill climbing plans: {breaks}.")


def plan_text(trades):
    """[(7, 21), (8, 22)] -> 'buy 7 sell 21, buy 8 sell 22' (hour 7 means 7-8 am)."""
    return ", ".join(f"buy {b} sell {s}" for b, s in trades) or "no trade"


def one_day(prices, date, node, battery, neighbors, use_z3):
    """Solve one day of real-time prices with every method and print each plan."""
    day = prices[(prices["date"] == date) & (prices["node"] == node)].sort_values("hour")
    if day["hour"].tolist() != list(range(24)):
        raise ValueError("Choose a date and hub with all 24 hours in data/prices.csv.")
    rt = day["rt_price"].tolist()
    count = count_plans(len(rt), battery)
    best_plan_cpsat(rt[:2], battery)  # untimed warm-up, so CP-SAT's start-up is not counted
    rows = []  # (method, plan, profit, milliseconds)
    if count <= BRUTE_LIMIT:
        (plan, profit), ms = timed(best_plan_brute, rt, battery)
        rows.append((f"Brute force, all {count:,} plans", plan, profit, ms))
    (plan, profit), ms = timed(best_plan_dp, rt, battery)
    rows.append(("Dynamic programming (exact)", plan, profit, ms))
    (plan, profit, status), ms = timed(best_plan_cpsat, rt, battery)
    rows.append((f"CP-SAT ({status})", plan, profit, ms))
    (plan, profit), ms = timed(hill_climb, rt, battery, neighbors)
    rows.append((f"Hill climbing ({neighbors})", plan, profit, ms))
    if use_z3:
        (plan, profit, status), ms = timed(best_plan_z3, rt, battery)
        rows.append((f"Z3 ({status})", plan, profit, ms))
    best = rows[0][2]  # brute force when it ran, otherwise dynamic programming
    trades = "trade" if battery.max_trades == 1 else "trades"
    heading(f"{hub_name(node)} on {date}, real-time prices: {battery.capacity_mwh:g} MWh "
            f"battery, up to {battery.max_trades} {trades} ($ per MW)")
    print(f"   {'Method':<32}{'Profit':>8}{'Of best':>9}{'Time':>11}   Plan (hour 7 = 7-8 am)")
    for method, plan, profit, ms in rows:
        share = f"{100 * profit / best:.1f}%" if best > 0 else "-"
        print(f"   {method:<32}{profit:>8.2f}{share:>9}{ms:>8.2f} ms   {plan_text(plan)}")
    breaks = sum(len(check_plan(plan, battery, len(rt))) for _, plan, _, _ in rows)
    print(f"   Checks: {len(rows)} plans replayed against the battery rules, {breaks} broken.")
    if count > BRUTE_LIMIT:
        print(f"   Brute force skipped: this day has {count:,} valid plans.")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--z3", action="store_true", help="also time Z3 (needs z3-solver)")
    parser.add_argument("--date", help="solve one day only, YYYY-MM-DD (needs --node)")
    parser.add_argument("--node", choices=HUBS)
    parser.add_argument("--capacity", type=float, default=2.0, help="MWh, one-day mode")
    parser.add_argument("--trades", type=int, default=2, help="most trades, one-day mode")
    parser.add_argument("--neighbors", choices=NEIGHBORS, default="split",
                        help="changes hill climbing may try in one-day mode")
    args = parser.parse_args()
    if (args.date is None) != (args.node is None):
        parser.error("One-day mode needs both --date YYYY-MM-DD and --node HUB_NAME.")
    if not 0.85 <= args.capacity <= 100 or not 1 <= args.trades <= 24:
        parser.error("Use a capacity from 0.85 to 100 MWh and 1 to 24 trades.")
    prices = pd.read_csv("data/prices.csv")
    if args.date is not None:
        battery = Battery(capacity_mwh=args.capacity, max_trades=args.trades)
        try:
            one_day(prices, args.date, args.node, battery, args.neighbors, args.z3)
        except ValueError as error:
            parser.error(str(error))
    else:
        Path("results").mkdir(exist_ok=True)
        daily = daily_comparison(prices)
        daily.to_csv("results/algorithms_daily.csv", index=False)
        print_daily(daily)
        horizons = horizon_comparison(prices, args.z3)
        horizons.to_csv("results/algorithms_horizons.csv", index=False)
        print_horizons(horizons)