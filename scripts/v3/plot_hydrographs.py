"""
Four stacked daily hydrographs for one water year, drawn like the app's hydrograph (app.py
plot_flows): observed flow, consumptive use stacked below zero, transbasin imports stacked above
zero, estimated natural flow (Q_NAT2) and USBR natural flow where it exists. Unlike the app,
both kinds of transbasin export (outside and within the CRB) are one category. Flows are on a log
axis; consumptive use (negative, stacked) and imports are in a linear strip under each hydrograph,
because a stack can't be read on a log axis (the layer nearest zero would fill the first decades).

    python scripts/v3/plot_hydrographs.py
    python scripts/v3/plot_hydrographs.py --wy 2018 --gages 09380000 09050700

Reads shiny-app/ucol_natural/timeseries/<gage>.parquet; writes plots/hydrographs_wy<year>.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
APP_DIR = REPO / "shiny-app" / "ucol_natural"
TS_DIR = APP_DIR / "timeseries"
OUT_DIR = REPO / "plots"

GAGES = ["09380000", "09291000", "09237450", "09073300"]

# Same colors and stacking order as app.py (first = nearest zero); interbasin + intrabasin
# exports are summed into "exports".
CU = {
    "irrigation": ("irrigation", "#7aab74", ["irrigation"]),
    "evap": ("reservoir evaporation", "#e6ab02", ["evap"]),
    "exports": ("transbasin exports", "#d95f02", ["interbasin", "intrabasin"]),
    "municipal": ("municipal", "#7570b3", ["municipal"]),
    "industrial": ("industrial", "#666666", ["industrial"]),
    "hydropower": ("hydropower", "#1b7837", ["hydropower"]),
}
IMPORT = ("transbasin imported", "#e7298a", "transbasin")


def load(gage, wy):
    df = pd.read_parquet(TS_DIR / f"{gage}.parquet")
    df = df.set_index(pd.to_datetime(df["date"])).loc[f"{wy - 1}-10-01":f"{wy}-09-30"]
    for key, (_, _, types) in CU.items():
        cols = [f"{t}_cfs" for t in types if f"{t}_cfs" in df.columns]
        df[f"cu_{key}"] = df[cols].sum(axis=1) if cols else 0.0
    return df


def plot_panel(ax, ax_cu, df):
    """Flows on ax (log), CU / imports on ax_cu (linear). Return {label: handle} for the legend."""
    handles = {}
    x = df.index
    stack = [k for k in CU if df[f"cu_{k}"].abs().sum() > 0]
    if stack:
        polys = ax_cu.stackplot(x, *[df[f"cu_{k}"] for k in stack],
                                colors=[CU[k][1] for k in stack], alpha=0.6, lw=0.5)
        handles.update({CU[k][0]: p for k, p in zip(stack, polys)})
    imp = f"{IMPORT[2]}_cfs"
    if imp in df.columns and df[imp].abs().sum() > 0:
        handles[IMPORT[0]] = ax_cu.fill_between(x, 0, df[imp], color=IMPORT[1], alpha=0.6, lw=0.5)
    ax_cu.axhline(0, color="gray", lw=0.6)

    handles["observed flow"], = ax.plot(x, df["Q_unfilled_cfs"], color="royalblue", lw=1.5)
    handles["estimated natural flow"], = ax.plot(x, df["Q_NAT2_cfs"], color="forestgreen", lw=2)
    if "Q_NATUSBR_cfs" in df.columns and df["Q_NATUSBR_cfs"].notna().any():
        handles["USBR estimated natural flow"], = ax.plot(x, df["Q_NATUSBR_cfs"], color="black",
                                                          lw=2, ls="--")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax_cu.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for a in (ax, ax_cu):
        a.grid(color="#f0f0f0", which="major")
        a.set_axisbelow(True)
    return handles


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wy", type=int, default=2020, help="water year (Oct 1 - Sep 30)")
    p.add_argument("--gages", nargs="+", default=GAGES)
    args = p.parse_args()

    info = pd.read_csv(TS_DIR / "basin_info.csv", dtype={"gage": str}).set_index("gage")
    n = len(args.gages)
    fig = plt.figure(figsize=(10, 3.4 * n))
    outer = fig.add_gridspec(n, 1, hspace=0.28)
    axes = []
    handles = {}
    for i, gage in enumerate(args.gages):
        inner = outer[i].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(inner[0], sharex=axes[0] if axes else None)
        ax_cu = fig.add_subplot(inner[1], sharex=ax)
        axes += [ax, ax_cu]
        handles.update(plot_panel(ax, ax_cu, load(gage, args.wy)))
        name = info.loc[gage, "name"] if gage in info.index else gage
        ax.set_title(f"({chr(97 + i)}) {name} (USGS-{gage})", loc="left", fontsize=11)
        ax.set_ylabel("Streamflow (cfs)")
        ax_cu.set_ylabel("CU (cfs)")
        ax.tick_params(labelbottom=False)
        if i < n - 1:
            ax_cu.tick_params(labelbottom=False)

    axes[-1].set_xlabel(f"Water year {args.wy}")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlim(pd.Timestamp(f"{args.wy - 1}-10-01"), pd.Timestamp(f"{args.wy}-09-30"))
    fig.align_ylabels(axes)

    order = ["observed flow", "estimated natural flow", "USBR estimated natural flow",
             IMPORT[0]] + [v[0] for v in CU.values()]
    labels = [k for k in order if k in handles]
    fig.legend([handles[k] for k in labels], labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.subplots_adjust(top=0.935, bottom=0.05)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"hydrographs_wy{args.wy}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
