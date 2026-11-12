import argparse
import pandas as pd
from backtest import backtest, business_summary, paired_tests, profit_statement, summarize
from prepare_data import HUBS
from report import heading, hub_name, print_report, usd
from trade import best_trade, realized


def recommend(prices, date, node, eff=0.85, wear=10):
    day = prices[(prices["date"] == date) & (prices["node"] == node)].sort_values("hour")
    if day["hour"].tolist() != list(range(24)):
        raise ValueError("Choose a date and hub with all 24 hours in data/prices.csv.")
    da, rt = day["da_price"].tolist(), day["rt_price"].tolist()
    buy, sell, expected = best_trade(da, eff, wear)
    actual = realized(buy, sell, rt, eff, wear)
    fixed = realized(13, 19, rt, eff, wear)
    c_buy, c_sell, hindsight = best_trade(rt, eff, wear)
    heading(f"Recommendation for {hub_name(node)} on {date} (Central time, per MW)")
    if buy is None:
        print("   B  day-ahead plan:     no trade (no profitable hours in the day-ahead prices)")
    else:
        print(f"   B  day-ahead plan:     charge {buy:02d}:00-{buy + 1:02d}:00, "
              f"sell {sell:02d}:00-{sell + 1:02d}:00")
        print(f"                          expected {usd(expected)}, actual {usd(actual)}")
    print(f"   A  fixed rule:         {usd(fixed)}  (buy 1 pm, sell 7 pm)")
    print(f"   C  perfect hindsight:  {usd(hindsight)}")
    print("\nModeled from historical prices, not actual battery earnings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["backtest", "recommend"])
    parser.add_argument("--date")
    parser.add_argument("--node", choices=HUBS)
    parser.add_argument("--eff", type=float, default=0.85)
    parser.add_argument("--wear", type=float, default=10)
    parser.add_argument("--mw", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.eff <= 1 or not 0 <= args.wear < float("inf"):
        parser.error("Use efficiency above 0 and at most 1, and a finite wear cost of 0 or more.")
    if not 0 < args.mw < float("inf"):
        parser.error("Use a battery size (--mw) above 0.")
    if args.command == "recommend" and (args.date is None or args.node is None):
        parser.error("recommend needs --date YYYY-MM-DD and --node HUB_NAME.")
    prices = pd.read_csv("data/prices.csv")
    if args.command == "backtest":
        daily = backtest(prices, args.eff, args.wear)
        print_report(daily, summarize(daily), paired_tests(daily), profit_statement(daily),
                     business_summary(daily, args.mw), args.eff, args.wear)
    else:
        recommend(prices, args.date, args.node, args.eff, args.wear)
