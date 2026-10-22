"""
Plot the leave-one-out test of the below-reservoir ungaged-inflow scaling.

Run naturalize.py first; it writes pipeline/data/inflow_validation.csv and
inflow_validation_by_year.csv. Each row there is one inflow gage, held out and
predicted from the other inflow gages of a below-reservoir basin.

    python pipeline/plot_inflow_validation.py
    python pipeline/plot_inflow_validation.py --hydrographs   # also daily obs vs predicted PDF

Outputs (pipeline/data/plots/)
  inflow_validation_scores.png    NSE and PBIAS per inflow gage, grouped by basin
  inflow_validation_share.png     NSE vs the gage's share of the basin's natural flow
  inflow_validation_yearly.png    per-water-year PBIAS of each inflow gage
  inflow_validation_hydrographs.pdf  (--hydrographs) one page per basin / inflow gage

The whole-record PBIAS is close to 0 by construction (the scaling ratio is fitted on
the same record), so the yearly PBIAS plot is the better view of volume error.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

BLUE, ORANGE = "#2a78d6", "#eb6834"        # observed / predicted (validated pair)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9,
})


def short(name, n=34):
    if not isinstance(name, str):
        return "?"
    # USGS names are upper case; title-case them but keep state codes ("..., CO")
    head, sep, tail = name.rpartition(", ")
    name = (head.title() + sep + tail) if sep and len(tail) == 2 else name.title()
    return name if len(name) <= n else name[:n - 1] + "…"


def load():
    for p in [C.INFLOW_VALIDATION_PATH, C.INFLOW_VALIDATION_YEARLY_PATH]:
        if not p.exists():
            raise SystemExit(f"{p} not found; run pipeline/naturalize.py first")
    dtype = {"gage": str, "inflow_gage": str}
    s = pd.read_csv(C.INFLOW_VALIDATION_PATH, dtype=dtype)
    y = pd.read_csv(C.INFLOW_VALIDATION_YEARLY_PATH, dtype=dtype)
    if s.empty:
        raise SystemExit("no inflow gages were evaluated (inflow_validation.csv is empty)")
    for d in (s, y):
        d["gage"] = d["gage"].apply(C.fix_gage_id)
        d["inflow_gage"] = d["inflow_gage"].apply(C.fix_gage_id)
    # one row per (basin, inflow gage): basins in name order, best NSE first within a basin
    s = s.sort_values(["name", "nse"], ascending=[True, False]).reset_index(drop=True)
    s["label"] = [f"{short(n)}  {g}  ({100 * sh:.0f}%)"
                  for n, g, sh in zip(s["inflow_name"], s["inflow_gage"], s["mean_share"])]
    return s, y


def basin_bands(ax, s, ypos, xtext):
    """Shade alternate basin groups and name each group at the top of its rows."""
    for i, (gname, grp) in enumerate(s.groupby("name", sort=False)):
        lo, hi = ypos[grp.index].min() - 0.5, ypos[grp.index].max() + 0.5
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#f1f0ec", zorder=0, lw=0)
        if xtext is not None:
            ax.text(xtext, lo + 0.1, f"{short(gname, 40)}  {grp['gage'].iloc[0]}",
                    transform=ax.get_yaxis_transform(), ha="left", va="top",
                    fontsize=8, fontweight="bold", color=INK)


def row_positions(s):
    """y position per row, with one blank row before each basin for its heading."""
    ypos, y = {}, 0
    for _, grp in s.groupby("name", sort=False):
        y += 1
        for i in grp.index:
            ypos[i] = y
            y += 1
    return pd.Series(ypos), y


def plot_scores(s, out):
    ypos, n = row_positions(s)
    fig, (a1, a2) = plt.subplots(1, 2, sharey=True, figsize=(11, 0.28 * n + 1.6),
                                 gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    for ax in (a1, a2):
        basin_bands(ax, s, ypos, None)
        ax.grid(axis="y", visible=False)

    nse = s["nse"].clip(lower=-1)
    a1.hlines(ypos, -1, nse, color=GRID, lw=1, zorder=1)
    a1.scatter(nse, ypos, s=36, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=3)
    low = s["nse"] < -1
    a1.scatter(np.full(low.sum(), -1), ypos[low], marker="<", s=40, color=BLUE, zorder=3)
    for v in (0, 0.5):
        a1.axvline(v, color=INK2, lw=0.8, ls=":" if v else "-", zorder=2)
    a1.set_xlim(-1.05, 1.0)
    a1.set_xlabel("NSE, daily  (◀ = below -1)")

    lim = max(10, np.ceil(s["pbias"].abs().max() / 5) * 5)
    a2.axvline(0, color=INK2, lw=0.8, zorder=2)
    a2.hlines(ypos, 0, s["pbias"], color=BLUE, lw=2, zorder=3)
    a2.scatter(s["pbias"], ypos, s=36, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=4)
    a2.set_xlim(-lim, lim)
    a2.set_xlabel("PBIAS, whole record (%)  (≈0 by construction)")

    a1.set_yticks(ypos.values, s["label"])
    a1.set_ylim(n, 0)
    # basin headings on the blank row above each group
    for gname, grp in s.groupby("name", sort=False):
        a1.text(-0.02, ypos[grp.index].min() - 0.75, f"{short(gname, 40)}  ({grp['gage'].iloc[0]})",
                transform=a1.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.5, fontweight="bold", color=INK)
    a1.tick_params(axis="y", length=0)
    fig.suptitle("Ungaged-inflow scaling, leave-one-out: each inflow gage predicted from the "
                 "others\n(% = mean share of the basin's natural flow in evaluated years)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_share(s, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(100 * s["mean_share"], s["nse"].clip(lower=-1), s=40, color=BLUE,
               edgecolor=SURFACE, lw=1.5, zorder=3)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(100 * 0.05, color=INK2, lw=0.8, ls=":")
    ax.text(100 * 0.05, 1.0, " 5% cutoff", va="top", fontsize=8, color=INK2)
    # label the weakest few so they can be found in the scores plot
    for _, r in s.nsmallest(min(5, len(s)), "nse").iterrows():
        ax.annotate(f"{r['inflow_gage']} @ {r['gage']}", (100 * r["mean_share"], max(r["nse"], -1)),
                    xytext=(5, -3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(4, 100)
    ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("held-out gage's mean share of basin natural flow (%, log)")
    ax.set_ylabel("NSE, daily (clipped at -1)")
    ax.set_ylim(-1.05, 1.02)
    ax.set_title("Does skill depend on how big the held-out piece is?", loc="left", fontsize=10.5)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_yearly(s, y, out):
    ypos, n = row_positions(s)
    key = dict(zip(zip(s["gage"], s["inflow_gage"]), s.index))
    y = y[y[["gage", "inflow_gage"]].apply(tuple, axis=1).isin(key)].copy()
    y["row"] = [ypos[key[(g, i)]] for g, i in zip(y["gage"], y["inflow_gage"])]
    lim = float(np.clip(np.nanpercentile(y["pbias"].abs(), 98) if len(y) else 50, 20, 200))

    fig, ax = plt.subplots(figsize=(8.5, 0.28 * n + 1.6))
    basin_bands(ax, s, ypos, None)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    for row, grp in y.groupby("row"):
        p = grp["pbias"].clip(-lim, lim)
        q1, med, q3 = np.nanpercentile(p, [25, 50, 75])
        ax.hlines(row, q1, q3, color=BLUE, lw=6, alpha=0.35, zorder=2)
        jitter = np.random.default_rng(int(row)).uniform(-0.18, 0.18, len(p))
        ax.scatter(p, row + jitter, s=10, color=BLUE, alpha=0.7, lw=0, zorder=3)
        ax.vlines(med, row - 0.32, row + 0.32, color=INK, lw=1.5, zorder=4)
    ax.set_yticks(ypos.values, s["label"])
    ax.tick_params(axis="y", length=0)
    for gname, grp in s.groupby("name", sort=False):
        ax.text(-0.02, ypos[grp.index].min() - 0.75, f"{short(gname, 40)}  ({grp['gage'].iloc[0]})",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8.5, fontweight="bold", color=INK)
    ax.set_ylim(n, 0)
    ax.set_xlim(-lim * 1.05, lim * 1.05)
    ax.set_xlabel(f"PBIAS per water year (%); dots = years, band = middle 50%, tick = median "
                  f"(clipped at ±{lim:.0f})")
    ax.set_title("Year-to-year volume error of the held-out prediction", loc="left", fontsize=10.5)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hydrographs(s, out):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out) as pdf:
        for gage, grp in s.groupby("gage", sort=False):
            path = C.NATURALIZED_FULL_DIR / f"{gage}.parquet"
            if not path.exists():
                print(f"  no {path.name}, skipping hydrographs for {gage}")
                continue
            df = pd.read_parquet(path).set_index("date")
            for _, r in grp.iterrows():
                ig = r["inflow_gage"]
                cols = [f"igage_{ig}", f"test_{ig}"]
                if not set(cols) <= set(df.columns):
                    continue
                d = df[cols].dropna(how="all")
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(d.index, d[cols[0]], color=BLUE, lw=1.2, label="observed (naturalized)")
                ax.plot(d.index, d[cols[1]], color=ORANGE, lw=1.2, label="predicted from other inflows")
                ax.set_ylabel("cfs")
                ax.legend(loc="upper left", frameon=False)
                ax.set_title(f"{short(r['inflow_name'], 60)} ({ig}) held out at "
                             f"{short(r['name'], 60)} ({gage})\nNSE {r['nse']:.2f}, "
                             f"PBIAS {r['pbias']:.1f}%, r {r['r']:.2f}, {r['n_years']} years, "
                             f"mean share {100 * r['mean_share']:.0f}%", loc="left", fontsize=10)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hydrographs", action="store_true",
                   help="also write a PDF of daily observed vs predicted for every inflow gage")
    args = p.parse_args()

    s, y = load()
    C.VALIDATION_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    d = C.VALIDATION_PLOT_DIR
    plot_scores(s, d / "inflow_validation_scores.png")
    plot_share(s, d / "inflow_validation_share.png")
    plot_yearly(s, y, d / "inflow_validation_yearly.png")
    if args.hydrographs:
        plot_hydrographs(s, d / "inflow_validation_hydrographs.pdf")

    print(f"{len(s)} inflow gages in {s['gage'].nunique()} basins; "
          f"median NSE {s['nse'].median():.2f}, {(s['nse'] < 0).sum()} below 0")
    print(f"plots in {d}")


if __name__ == "__main__":
    main()
