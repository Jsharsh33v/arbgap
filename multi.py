"""Several trades a day: exact solvers and a faster heuristic.

A plan is a list of trades (buy_hour, sell_hour), the same format check_plan uses.
Each trade buys 1 MWh in one hour (1 MW for one hour) and later sells the eff MWh
it stored. The battery holds at most `units(battery)` trades' worth of energy at once,
makes at most `battery.max_trades` trades a day, and starts and ends the day empty.
"""
from ortools.sat.python import cp_model
from trade import trade_profit

SCALE = 1000  # CP-SAT needs whole numbers, so money is counted in tenths of a cent


def units(battery):
    """How many charged hours fit in the battery (1 MWh bought stores eff MWh)."""
    return int(battery.capacity_mwh / (battery.eff * battery.power_mw) + 1e-9)


def plan_profit(trades, prices, battery):
    """Profit of a plan: the sum of its trades, minus the cost per trade."""
    return sum(trade_profit(prices[b], prices[s], battery.eff, battery.wear)
               - battery.trade_cost for b, s in trades)


def to_trades(actions):
    """Hourly actions (1 buy, -1 sell, 0 wait) -> (buy, sell) pairs, oldest energy first."""
    waiting, trades = [], []
    for hour, action in enumerate(actions):
        if action == 1:
            waiting.append(hour)
        elif action == -1:
            trades.append((waiting.pop(0), hour))
    return trades


def best_plan_brute(prices, battery):
    """Exact, tiny days only: tries every hour-by-hour plan (up to 3^H of them)."""
    most, limit = units(battery), battery.max_trades
    best = ([], 0.0)

    def search(hour, level, used, actions):
        nonlocal best
        if hour == len(prices):
            if level == 0:
                trades = to_trades(actions)
                profit = plan_profit(trades, prices, battery)
                if profit > best[1] + 1e-9:
                    best = (trades, profit)
            return
        search(hour + 1, level, used, actions + [0])
        if level < most and used < limit:
            search(hour + 1, level + 1, used + 1, actions + [1])
        if level > 0:
            search(hour + 1, level - 1, used, actions + [-1])

    search(0, 0, 0, [])
    return best


def best_plan_dp(prices, battery):
    """Exact: dynamic programming over (hour, energy stored, trades used).

    Keeps the best profit for every battery state after each hour, so the work is
    O(H x units x max_trades) instead of trying every plan.
    """
    most, limit = units(battery), battery.max_trades
    buy = [-p - battery.trade_cost for p in prices]
    sell = [battery.eff * (p - battery.wear) for p in prices]
    best = {(0, 0): 0.0}  # (energy stored, trades used) -> best profit so far
    steps = []  # for each hour: new state -> (old state, action)
    for hour in range(len(prices)):
        new, step = {}, {}
        for (level, used), value in best.items():
            options = [((level, used), value, 0)]  # waiting comes first, so ties mean no trade
            if level < most and used < limit:
                options.append(((level + 1, used + 1), value + buy[hour], 1))
            if level > 0:
                options.append(((level - 1, used), value + sell[hour], -1))
            for state, total, action in options:
                if state not in new or total > new[state] + 1e-9:
                    new[state], step[state] = total, ((level, used), action)
        best = new
        steps.append(step)
    state = max((s for s in best if s[0] == 0), key=lambda s: (best[s], -s[1]))
    profit = best[state]
    actions = []
    for step in reversed(steps):
        state, action = step[state]
        actions.append(action)
    return to_trades(actions[::-1]), profit


def best_plan_cpsat(prices, battery, time_limit=None, gap=None):
    """Google OR-Tools CP-SAT. Exact with no limits; with a time or gap limit it
    returns the best plan found so far (FEASIBLE = valid but not proven best)."""
    hours, most = len(prices), units(battery)
    model = cp_model.CpModel()
    buy = [model.NewBoolVar(f"buy_{t}") for t in range(hours)]
    sell = [model.NewBoolVar(f"sell_{t}") for t in range(hours)]
    level = [model.NewIntVar(0, most, f"level_{t}") for t in range(hours)]
    for t in range(hours):
        model.Add(buy[t] + sell[t] <= 1)  # one action per hour (power limit)
        before = level[t - 1] if t > 0 else 0
        model.Add(level[t] == before + buy[t] - sell[t])  # energy stored, never below 0
    model.Add(level[-1] == 0)  # end the day empty
    model.Add(sum(buy) <= battery.max_trades)
    money_buy = [round((-p - battery.trade_cost) * SCALE) for p in prices]
    money_sell = [round(battery.eff * (p - battery.wear) * SCALE) for p in prices]
    model.Maximize(sum(money_buy[t] * buy[t] + money_sell[t] * sell[t] for t in range(hours)))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1  # one core, so timings are fair
    if time_limit is not None:
        solver.parameters.max_time_in_seconds = time_limit
    if gap is not None:
        solver.parameters.relative_gap_limit = gap
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], 0.0, solver.StatusName(status)
    actions = [1 if solver.Value(buy[t]) else -1 if solver.Value(sell[t]) else 0
               for t in range(hours)]
    trades = to_trades(actions)
    return trades, plan_profit(trades, prices, battery), solver.StatusName(status)


def best_new_trade(prices, battery, trades):
    """The most profitable trade that can be added to a plan without breaking a rule."""
    hours, most = len(prices), units(battery)
    if len(trades) >= battery.max_trades:
        return None, 0.0
    busy, level = [False] * hours, [0] * hours
    for b, s in trades:
        busy[b] = busy[s] = True
        for t in range(b, s):
            level[t] += 1
    best, gain = None, 0.0
    for b in range(hours):
        if busy[b] or level[b] >= most:
            continue
        for s in range(b + 1, hours):
            if level[s - 1] >= most:
                break  # the battery would be over capacity at hour s - 1
            if busy[s]:
                continue
            value = trade_profit(prices[b], prices[s], battery.eff, battery.wear)
            value -= battery.trade_cost
            if value > gain + 1e-9:
                best, gain = (b, s), value
    return best, gain


def hill_climb(prices, battery):
    """Heuristic: start with no trades and keep making the one change that adds the
    most profit (add a trade, or swap one trade for a better one) until nothing helps.

    Every plan it builds is valid, but it can stop at a local optimum: a plan that no
    single change improves even though a better plan exists.
    """
    trades = []
    while True:
        move, gain = None, 1e-9
        new, value = best_new_trade(prices, battery, trades)
        if new is not None and value > gain:
            move, gain = trades + [new], value
        for i, old in enumerate(trades):
            others = trades[:i] + trades[i + 1:]
            new, value = best_new_trade(prices, battery, others)
            old_value = plan_profit([old], prices, battery)
            if new is not None and value - old_value > gain:
                move, gain = others + [new], value - old_value
        if move is None:
            return sorted(trades), plan_profit(trades, prices, battery)
        trades = move