"""
Median water-year hydrographs of estimated natural flow (Q_NAT2) and USBR natural flow at the 10
largest rivers with USBR data, one gage per river (the one with the highest mean USBR flow, so
Lees Ferry for the Colorado, Green River UT for the Green), largest first, as a 5 x 2 grid.

    python scripts/v3/usbr_median_hydrographs.py
    python scripts/v3/usbr_median_hydrographs.py --n 6 --out some/other.png

Daily values for both datasets, no monthly resampling. USBR is published monthly and spread to
daily, so its line is stepped by month. A water year is used only when both Q_NAT2_cfs and
Q_NATUSBR_cfs exist on every day of it. Feb 29 is dropped so every year has 365 days. Each panel
shows the median by day of water year, with the 25th-75th percentile range lightly shaded.

Reads  shiny-app/ucol_natural/timeseries/<gage>.parquet and basin_info.csv (names)
Writes plots/usbr_median_hydrographs.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TS_DIR = REPO / "shiny-app" / "ucol_natural" / "timeseries"
OUT_DIR = REPO / "plots"

SERIES = [  # column, label, color, line style (same colors as the app)
    ("Q_NAT2_cfs", "Estimated natural flow", "forestgreen", "-"),
    ("Q_NATUSBR_cfs", "USBR natural flow", "black", "--"),
]
MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
MONTH_START = np.cumsum([0, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31])  # day-of-WY index


def load(gage):
    """Daily Q_NAT2 and USBR for complete water years where both exist, with wy and day-of-WY."""
    df = pd.read_parquet(TS_DIR / f"{gage}.parquet", columns=["date"] + [s[0] for s in SERIES])
    df = df.set_index(pd.to_datetime(df["date"])).drop(columns="date")
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    df["wy"] = df.index.year + (df.index.month >= 10)
    ok = df.groupby("wy")[[s[0] for s in SERIES]].apply(lambda g: len(g) == 365 and g.notna().all().all())
    df = df[df["wy"].isin(ok[ok].index)].copy()
    df["dowy"] = df.groupby("wy").cumcount()
    return df


def river(name):
    """River name from a gage name: 'Green River Near Greendale, UT' -> 'Green'."""
    return name.split(" River")[0].strip()


def biggest_usbr_gages(n, names):
    """The n gages with the highest mean USBR natural flow, one per river (the biggest gage on
    each river, e.g. Lees Ferry for the Colorado)."""
    means = {}
    for path in TS_DIR.glob("*.parquet"):
        q = pd.read_parquet(path, columns=["Q_NATUSBR_cfs"])["Q_NATUSBR_cfs"]
        if q.notna().any():
            means[path.stem] = q.mean()
    picked, rivers = [], set()
    for gage in sorted(means, key=means.get, reverse=True):
        r = river(names.get(gage, gage))
        if r not in rivers and len(load(gage)):  # skip gages with no complete shared water year
            picked.append(gage)
            rivers.add(r)
    return picked[:n]


def plot_panel(ax, df):
    for col, label, color, ls in SERIES:
        q = df.pivot(index="dowy", columns="wy", values=col).quantile([0.25, 0.5, 0.75], axis=1).T
        ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.12, lw=0)
        ax.plot(q.index, q[0.5], color=color, ls=ls, lw=1.6, label=label)
    ax.set_xlim(0, 364)
    ax.set_ylim(bottom=0)
    ax.set_xticks(MONTH_START)
    ax.set_xticklabels(MONTHS, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(color="#e8e8e8", lw=0.7)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10, help="number of gages (2 columns)")
    p.add_argument("--out", type=Path, default=OUT_DIR / "usbr_median_hydrographs.png")
    args = p.parse_args()

    names = pd.read_csv(TS_DIR / "basin_info.csv", dtype={"gage": str}).set_index("gage")["name"]
    gages = biggest_usbr_gages(args.n, names)
    nrows = int(np.ceil(len(gages) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 2.6 * nrows), sharex=True)
    axes = np.atleast_2d(axes)
    for i, (ax, gage) in enumerate(zip(axes.flat, gages)):
        df = load(gage)
        wys = df["wy"].unique()
        plot_panel(ax, df)
        ax.set_title(f"({chr(97 + i)}) {names.get(gage, gage)} (USGS-{gage})", loc="left", fontsize=9.5)
        ax.text(0.99, 0.95, f"WY {wys.min()}-{wys.max()}, n = {len(wys)}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#555555")
        print(f"{gage}  {len(wys)} water years ({wys.min()}-{wys.max()})")
    for ax in axes.flat[len(gages):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Streamflow (cfs)", fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.2))
    labels.append("Interquartile range")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Median water-year hydrographs: estimated vs. USBR natural flow", y=1.025, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
