from dataclasses import dataclass


@dataclass
class Battery:
    """Battery settings. The defaults are the current model."""
    power_mw: float = 1.0  # most it can buy or sell in one hour
    capacity_mwh: float = 1.0  # most energy it can hold
    eff: float = 0.85  # share of bought energy that ends up stored
    wear: float = 10.0  # $ per MWh sold
    trade_cost: float = 0.0  # $ per trade (one buy + one sell)
    max_trades: int = 1  # trades allowed per day


def check_plan(trades, battery, hours=24):
    """Replay one day's plan hour by hour and list every rule it breaks.

    trades is a list of (buy_hour, sell_hour) pairs. Each trade buys at full power
    for one hour and later sells everything that buy stored. No problems = valid plan.
    """
    problems = []
    if len(trades) > battery.max_trades:
        problems.append(f"{len(trades)} trades, but only {battery.max_trades} allowed")
    charge = [0.0] * hours  # MWh bought in each hour
    discharge = [0.0] * hours  # MWh sold in each hour
    for buy, sell in trades:
        if not 0 <= buy < sell < hours:
            problems.append(f"trade ({buy}, {sell}) must buy first, inside hours 0-{hours - 1}")
            continue
        charge[buy] += battery.power_mw
        discharge[sell] += battery.eff * battery.power_mw
    level = 0.0  # MWh stored in the battery
    for hour in range(hours):
        if charge[hour] > battery.power_mw or discharge[hour] > battery.power_mw:
            problems.append(f"hour {hour}: more than {battery.power_mw} MW")
        if charge[hour] > 0 and discharge[hour] > 0:
            problems.append(f"hour {hour}: buys and sells in the same hour")
        level += battery.eff * charge[hour] - discharge[hour]
        if level > battery.capacity_mwh + 1e-9:
            problems.append(f"hour {hour}: {level:.2f} MWh is over capacity "
                            f"({battery.capacity_mwh} MWh)")
        if level < -1e-9:
            problems.append(f"hour {hour}: battery level below zero")
    return problems 