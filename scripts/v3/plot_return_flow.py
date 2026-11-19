"""
Irrigation diversion (solid), consumptive use (shaded under the diversion) and lagged return flow
(dotted) for two irrigation-dominated basins with different irrigated-land-to-stream distances
(fast and slow return-flow response), for one water year: one panel per basin (cfs), plus the
Glover response of each basin.

Return flow is rebuilt from the published timeseries with the same algebra as naturalize.py:
    diversion   = -irrigation_cfs / efficiency          (efficiency = irr_cu_factor)
    CU          = -irrigation_cfs
    return flow = Glover-lagged (1 - efficiency) * diversion
The Glover kernel (naturalize.glover_unit_response, T = 5000 ft2/d, S = 0.2, normalized, 3-year
warm-up) is convolved over the full record, then the water year is cut out, so return flow from
earlier years carries in. The response speed is set by the irrigated-land-to-stream distance a:
the kernel peaks near a^2 S / (6 T) days.

    python scripts/v3/plot_return_flow.py
    python scripts/v3/plot_return_flow.py --wy 2018 --gages 09237450 09301500

Reads  shiny-app/ucol_natural/timeseries/<gage>.parquet and basin_info.csv
Writes plots/return_flow_wy<year>.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
APP_DIR = REPO / "shiny-app" / "ucol_natural"
TS_DIR = APP_DIR / "timeseries"
OUT_DIR = REPO / "plots"
sys.path.insert(0, str(APP_DIR / "pipeline"))
from naturalize import (GLOVER_MAX_LAG_DAYS, GLOVER_S, GLOVER_T,  # noqa: E402
                        apply_glover_warmup, glover_unit_response)

FT_TO_M = 0.3048
# Yampa above Stagecoach Reservoir (short distance, fast), Uinta at Randlett (long distance, slow)
GAGES = ["09237450", "09301500"]
COLORS = ["#1f77b4", "#7570b3"]
CU_ALPHA = 0.25   # consumptive use: shaded from 0 up to CU, under the diversion line


def kernel(a_ft):
    k = glover_unit_response(np.arange(GLOVER_MAX_LAG_DAYS), a_ft, GLOVER_T, GLOVER_S)
    return k / k.sum()


def irrigation_terms(gage, eff, a_ft):
    """Daily irrigation diversion, CU and lagged return flow (all positive, cfs), full record."""
    df = pd.read_parquet(TS_DIR / f"{gage}.parquet", columns=["date", "irrigation_cfs"])
    df = df.set_index(pd.to_datetime(df["date"]))
    cu = -df["irrigation_cfs"]
    div = cu / eff
    rf = (div * (1 - eff)).fillna(0).values
    return pd.DataFrame({"div": div, "cu": cu, "rf": apply_glover_warmup(rf, kernel(a_ft))},
                        index=df.index)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wy", type=int, default=2020, help="water year (Oct 1 - Sep 30)")
    p.add_argument("--gages", nargs="+", default=GAGES, help=f"up to {len(COLORS)} basins")
    p.add_argument("--out", type=Path, help="output PNG (default plots/return_flow_wy<year>.png)")
    args = p.parse_args()
    gages = args.gages[:len(COLORS)]

    info = pd.read_csv(TS_DIR / "basin_info.csv", dtype={"gage": str}).set_index("gage")
    n = len(gages)
    fig = plt.figure(figsize=(14, 3.2 * n + 0.6))
    grid = fig.add_gridspec(n, 2, width_ratios=[2.6, 1], hspace=0.35, wspace=0.18)
    axk = fig.add_subplot(grid[:, 1])
    start, end = pd.Timestamp(f"{args.wy - 1}-10-01"), pd.Timestamp(f"{args.wy}-09-30")
    lag_days = np.arange(GLOVER_MAX_LAG_DAYS)
    axes = []

    for i, (gage, color) in enumerate(zip(gages, COLORS)):
        ax = fig.add_subplot(grid[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        r = info.loc[gage]
        eff, a_ft = r["irr_cu_factor"], r["glover_a_ft"]
        d = irrigation_terms(gage, eff, a_ft).loc[start:end]
        h_cu = ax.fill_between(d.index, 0, d["cu"], color=color, alpha=CU_ALPHA, lw=0)
        h_div, = ax.plot(d.index, d["div"], color=color, lw=1.8)
        h_rf, = ax.plot(d.index, d["rf"], color=color, lw=1.8, ls=":")
        ax.legend([h_div, h_cu, h_rf], ["irrigation diversion", "consumptive use", "return flow"],
                  loc="upper left", fontsize=8.5, framealpha=0.9)
        name = r["name"].split(",")[0]
        ax.set_title(f"({chr(97 + i)}) {name} (USGS-{gage}): {a_ft * FT_TO_M:,.0f} m "
                     f"({a_ft:,.0f} ft) irrigated land to stream", loc="left", fontsize=10.5)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Flow (cfs)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(color="#e8e8e8")
        ax.set_axisbelow(True)
        if i < n - 1:
            ax.tick_params(labelbottom=False)

        k = kernel(a_ft)
        axk.plot(lag_days, k * 100, color=color, lw=1.8, label=name)
        axk.annotate(f"peak {lag_days[k.argmax()]} d", (lag_days[k.argmax()], k.max() * 100),
                     xytext=(6, 2), textcoords="offset points", fontsize=8.5, color=color)
        print(f"{gage}  eff {eff:.3f}  a {a_ft:.0f} ft  WY{args.wy}: diversion {d['div'].mean():.1f}, "
              f"CU {d['cu'].mean():.1f}, return flow {d['rf'].mean():.1f} cfs (mean); "
              f"kernel peak day {k.argmax()}")

    axes[-1].set_xlim(start, end)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1].set_xlabel(f"Water year {args.wy}")

    axk.set_xlim(0, 365)
    axk.set_ylim(bottom=0)
    axk.set_xlabel("Days after water is applied")
    axk.set_ylabel("Share returned per day (%)")
    axk.set_title(f"({chr(97 + n)}) Glover return-flow response", loc="left", fontsize=10.5)
    axk.grid(color="#e8e8e8")
    axk.set_axisbelow(True)

    out = args.out or OUT_DIR / f"return_flow_wy{args.wy}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
