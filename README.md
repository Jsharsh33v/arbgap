# ArbGap Core

Compare one daily electricity trade at four ERCOT hubs:

- **A:** buy at hour 13 and sell at hour 19 every day.
- **B:** choose hours with day-ahead prices; evaluate the trade at real-time prices.
- **C:** choose hours with real-time prices known afterward (perfect hindsight).

Set up from this folder:

```bash
cd ~/Documents/arbgap_core
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
```

Download and unzip the 2026 files into `data/raw/`:
[NP4-180-ER (day-ahead)](https://www.ercot.com/mp/data-products/data-product-details?id=NP4-180-ER) and
[NP6-785-ER (real-time)](https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-ER).
The included May-July 2026 data have 92 days and 8,832 hourly rows across
HB_HOUSTON, HB_NORTH, HB_SOUTH, and HB_WEST. Hour 13 means 1-2 p.m. Central time.
If 2026 is incomplete, download both 2025 files and prepare with `--year 2025`.

```bash
python prepare_data.py
python cli.py backtest
python cli.py recommend --date 2026-07-01 --node HB_HOUSTON
python cli.py backtest --wear 20
python -m pytest -q
```

For the July 1 Houston demo (default settings), B charges 8-9 a.m. and sells 8-9 p.m.
Expected value is $14.04; realized B is $11.60. A loses $7.99; hindsight C makes $15.97.
All are modeled dollars per MW for that day. July 15 Houston gives no trade.
Across the 92 dates, after averaging the four hubs, B beats A on 78% of days
and earns $6.64 more per MW per day on average. These are descriptive results.

`prepare_data.py` averages four RT intervals per hour and saves `data/prices.csv`.
It rejects duplicates and prints incomplete dates it drops. All four hubs stay together.
`trade.py` calculates `eff * (sell_price - wear) - buy_price` and tries every legal hour pair.
`backtest.py` compares A/B/C; `cli.py` handles the commands above.
Results go to `results/daily.csv`, `results/summary.csv`, and `results/mean_values.png`.
The daily file includes B's `b_buy` and `b_sell` hours; blank means no trade.
In Houston, B starts at 8, 9, or 10 a.m. on 72 of its 90 trading days.
A loses on 58 of 92 Houston days but averages +$0.40 per MW per day overall.
The timing pattern alone does not explain the full difference between the strategies.
The summary gives mean, median, and percent above zero for B-A and C-B, overall and per hub.
Overall results average the four hubs per date first. Each backtest replaces the results.

Assumptions: buy 1 MWh (1 MW for one hour), sell 0.85 MWh, and pay $10 wear per MWh sold.
Efficiency and wear are adjustable assumptions. B/C can skip a trade; A always trades.
These are modeled energy-market values, not actual company earnings or significance tests.
C is an upper bound in this one-trade model. Detailed battery limits and other costs are left out.
