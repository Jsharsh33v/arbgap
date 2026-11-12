def trade_profit(p_buy, p_sell, eff=0.85, wear=10):
    """Buy 1 MWh, then sell eff MWh after energy losses."""
    return eff * (p_sell - wear) - p_buy


def best_trade(prices, eff=0.85, wear=10):
    """Try every buy hour and every later sell hour."""
    best_buy = None
    best_sell = None
    best_profit = 0.0

    for buy in range(len(prices)):
        for sell in range(buy + 1, len(prices)):
            profit = trade_profit(prices[buy], prices[sell], eff, wear)
            if profit > best_profit:
                best_buy = buy
                best_sell = sell
                best_profit = profit

    return best_buy, best_sell, best_profit


def realized(buy, sell, prices, eff=0.85, wear=10):
    """Use the chosen hours without changing the plan."""
    if buy is None and sell is None:
        return 0.0
    return trade_profit(prices[buy], prices[sell], eff, wear)


def trade_breakdown(p_buy, p_sell, eff=0.85, wear=10):
    """Split one trade into money in and money out ($ per MW of battery).

    Buy 1 MWh at p_buy, sell eff MWh at p_sell, and pay wear on every MWh sold.
    """
    revenue = eff * p_sell
    energy_cost = p_buy
    wear_cost = eff * wear
    profit = revenue - energy_cost - wear_cost
    return {"revenue": revenue, "energy_cost": energy_cost,
            "wear_cost": wear_cost, "profit": profit}


def realized_breakdown(buy, sell, prices, eff=0.85, wear=10):
    """Breakdown for the chosen hours. A no-trade day is all zeros."""
    if buy is None and sell is None:
        return {"revenue": 0.0, "energy_cost": 0.0, "wear_cost": 0.0, "profit": 0.0}
    return trade_breakdown(prices[buy], prices[sell], eff, wear)
