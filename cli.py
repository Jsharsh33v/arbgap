import argparse
import pandas as pd
from backtest import backtest, summarize
from prepare_data import HUBS
from trade import best_trade, realized


def recommend(prices, date, node, eff=0.85, wear=10):
    day = prices[(prices["date"] == date) & (prices["node"] == node)].sort_values("hour")
    if day["hour"].tolist() != list(range(24)):
        raise ValueError("Choose a date and hub with all 24 hours in data/prices.csv.")
    buy, sell, expected = best_trade(day["da_price"].tolist(), eff, wear)
    actual = realized(buy, sell, day["rt_price"].tolist(), eff, wear)
    print(f"{date} / {node} (Central time)")
    if buy is None:
        print("No trade: no positive day-ahead opportunity.")
    else:
        print(f"Charge: {buy:02d}:00-{buy + 1:02d}:00; sell: {sell:02d}:00-{sell + 1:02d}:00")
    print(f"Expected: ${expected:.2f}/MW; realized: ${actual:.2f}/MW for the day.")
    fixed = realized(13, 19, day["rt_price"].tolist(), eff, wear)
    c_buy, c_sell, hindsight = best_trade(day["rt_price"].tolist(), eff, wear)
    print(f"Fixed rule A: ${fixed:.2f}/MW; hindsight C: ${hindsight:.2f}/MW for the day.")
    print("Modeled value from historical prices, not actual battery earnings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["backtest", "recommend"])
    parser.add_argument("--date")
    parser.add_argument("--node", choices=HUBS)
    parser.add_argument("--eff", type=float, default=0.85)
    parser.add_argument("--wear", type=float, default=10)
    args = parser.parse_args()
    if not 0 < args.eff <= 1 or not 0 <= args.wear < float("inf"):
        parser.error("Use efficiency above 0 and at most 1, and a finite wear cost of 0 or more.")
    if args.command == "recommend" and (args.date is None or args.node is None):
        parser.error("recommend needs --date YYYY-MM-DD and --node HUB_NAME.")
    prices = pd.read_csv("data/prices.csv")
    if args.command == "backtest":
        daily = backtest(prices, args.eff, args.wear)
        summarize(daily)
    else:
        recommend(prices, args.date, args.node, args.eff, args.wear)
