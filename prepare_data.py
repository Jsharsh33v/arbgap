from pathlib import Path
import argparse
import pandas as pd

HUBS = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"]
KEYS = ["date", "node", "hour"]


def read_prices(market, year):
    paths = list(Path("data/raw").rglob(f"*{market}LZHBSPP_{year}*.xlsx"))
    if len(paths) != 1:
        raise ValueError(f"Put one {market} workbook for {year} in data/raw.")
    sheets = pd.read_excel(paths[0], sheet_name=None)
    frames = []
    for name, sheet in sheets.items():
        print(market, name, list(sheet.columns), flush=True)
        if not sheet.empty:
            frames.append(sheet)
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.rename(columns={
        "Delivery Date": "date", "Settlement Point": "node",
        "Settlement Point Name": "node", "Hour Ending": "hour",
        "Delivery Hour": "hour", "Settlement Point Price": "price",
    })
    prices["date"] = pd.to_datetime(prices["date"], format="mixed")
    summer = prices["date"].between(f"{year}-05-01", f"{year}-07-31")
    prices = prices[summer & prices["node"].isin(HUBS)].copy()
    if market == "DAM":
        prices["hour"] = prices["hour"].str.split(":").str[0].astype(int) - 1
    else:
        prices["hour"] = prices["hour"] - 1
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices["price"] = prices["price"].replace([float("inf"), -float("inf")], float("nan"))
    keys = KEYS.copy()
    if market == "RTM":
        keys.append("Delivery Interval")
        if not prices["Delivery Interval"].isin([1, 2, 3, 4]).all():
            raise ValueError("RT intervals must be 1, 2, 3, or 4.")
    if prices.duplicated(keys).any() or not prices["hour"].isin(range(24)).all():
        raise ValueError("Check the source for duplicate records or invalid hours.")
    return prices


def prepare_data(year=2026):
    da = read_prices("DAM", year).rename(columns={"price": "da_price"})
    rt = read_prices("RTM", year)
    # Each hourly average needs four valid, distinct intervals.
    rt = rt.groupby(KEYS)["price"].agg(["mean", "count"]).reset_index()
    rt = rt[rt["count"] == 4].rename(columns={"mean": "rt_price"})
    prices = da[KEYS + ["da_price"]].merge(rt[KEYS + ["rt_price"]], on=KEYS)

    complete_days = []
    dates = pd.date_range(f"{year}-05-01", f"{year}-07-31")
    for date in dates:
        day = prices[prices["date"] == date]
        # Keep all four hubs together: 4 hubs times 24 hours = 96 rows.
        if len(day) == 96 and not day[["da_price", "rt_price"]].isna().any().any():
            complete_days.append(day)
        else:
            print("Dropped", date.date(), "because a hub has missing hours or prices.")
    if not complete_days:
        raise ValueError("No complete dates. Check the files or try --year 2025.")
    prices = pd.concat(complete_days).sort_values(KEYS)
    prices.to_csv("data/prices.csv", index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(prices)} rows. Dropped {len(dates) - len(complete_days)} dates.")
    return prices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()
    prepare_data(args.year)
