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
