"""Charts for the Econ results. Run after the backtest and sensitivity.py:

    python cli.py backtest
    python sensitivity.py
    python charts.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {"B_minus_A": "Better decisions (B - A)", "C_minus_B": "Better information (C - B)"}


def gaps_by_hub(daily):
    """Average value of better decisions and better information at each hub."""
    by_date = daily.groupby("date")[["B_minus_A", "C_minus_B"]].mean()
    means = daily.groupby("node")[["B_minus_A", "C_minus_B"]].mean()
    means.loc["All hubs"] = by_date.mean()
    ax = means.rename(columns=LABELS).plot.bar(rot=0, figsize=(8, 4.5))
    ax.set_xlabel("")
    ax.set_ylim(0, means.max().max() * 1.3)  # room for the legend above the bars
    ax.set_ylabel("$ per MW per day")
    ax.set_title("Value of better decisions vs. better information, May-July 2026")
    plt.tight_layout()
    plt.savefig("results/gaps_by_hub.png", dpi=150)
    plt.close()


def gaps_over_time(daily):
    """Daily values (four hubs averaged) with a 7-day average so the trend is readable."""
    by_date = daily.groupby("date")[["B_minus_A", "C_minus_B"]].mean()
    dates = pd.to_datetime(by_date.index)
    weekly = by_date.rolling(7).mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for column, label in LABELS.items():
        ax.plot(dates, weekly[column], label=label)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend()
    ax.set_xlabel("")
    ax.set_ylabel("$ per MW per day (7-day average)")
    ax.set_title("Value of better decisions and better information over time")
    plt.tight_layout()
    plt.savefig("results/gaps_over_time.png", dpi=150)
    plt.close()


def sensitivity_chart(table):
    """How the two values change with battery efficiency, one line per wear cost."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, column in zip(axes, ["B_minus_A", "C_minus_B"]):
        for wear, group in table.groupby("wear"):
            ax.plot(group["eff"] * 100, group[column], marker="o", label=f"wear ${wear:g}/MWh")
        ax.set_title(LABELS[column])
        ax.set_xlabel("Efficiency (%)")
    axes[0].set_ylabel("$ per MW per day")
    axes[1].legend()
    fig.suptitle("Sensitivity to battery assumptions")
    plt.tight_layout()
    plt.savefig("results/sensitivity.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    daily = pd.read_csv("results/daily.csv")
    gaps_by_hub(daily)
    gaps_over_time(daily)
    sensitivity_chart(pd.read_csv("results/sensitivity.csv"))
    print("Saved gaps_by_hub.png, gaps_over_time.png and sensitivity.png in results/")