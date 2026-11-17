"""
Plot the leave-one-out test of the below-reservoir ungaged-inflow scaling over time.

Run naturalize.py first; it writes pipeline/data/inflow_validation_by_year.csv (one daily NSE
per basin, held-out inflow gage and water year) and the daily held-out series (igage_<gage>
observed, test_<gage> predicted from the other inflow gages) in
pipeline/data/naturalized_full/<basin>.parquet.

    python pipeline/plot_inflow_validation.py
    python pipeline/plot_inflow_validation.py --hydrographs   # also daily obs vs predicted PDF

Only water years in which the held-out gage carries between C.MIN_VALIDATION_SHARE and
C.MAX_VALIDATION_SHARE (common.py) of the basin's estimated natural flow are used, limited to
water years WATER_YEARS and to inflow gages with at least MIN_GAGE_YEARS such years. The filter
is applied here too, so tightening the bounds needs no naturalize.py rerun (loosening the
minimum below the value naturalize.py ran with has no effect: those years were never predicted).

Outputs (pipeline/data/plots/)
  inflow_validation_by_year.png       left: median NSE (top) and percent bias (bottom) per water
                                      year, interquartile range shaded; right: map of held-out
                                      inflow gages colored by median yearly NSE (basemap tiles
                                      need contextily and internet; skipped otherwise)
  inflow_validation_by_year_stats.csv numbers behind the left panels
  inflow_validation_by_gage_stats.csv numbers behind the map
  inflow_validation_hydrographs.pdf   (--hydrographs) one page per basin / inflow gage
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


WATER_YEARS = (1980, 2026)   # complete water years only (WY2027 is a few October weeks)
MIN_GAGE_YEARS = 5           # drop held-out inflow gages scored in fewer water years


def load():
    if not C.INFLOW_VALIDATION_YEARLY_PATH.exists():
        raise SystemExit(f"{C.INFLOW_VALIDATION_YEARLY_PATH} not found; run pipeline/naturalize.py first")
    y = pd.read_csv(C.INFLOW_VALIDATION_YEARLY_PATH, dtype={"gage": str, "inflow_gage": str})
    for col in ("gage", "inflow_gage"):
        y[col] = y[col].apply(C.fix_gage_id)
    y = y[y["share"].between(C.MIN_VALIDATION_SHARE, C.MAX_VALIDATION_SHARE)]
    y = y[y["water_year"].between(*WATER_YEARS)].dropna(subset=["nse"])
    n = y.groupby("inflow_gage")["water_year"].transform("size")
    return y[n >= MIN_GAGE_YEARS]


METRICS = {"nse": "NSE", "pbias": "Percent bias (%)"}


def yearly_stats(y):
    g = y.groupby("water_year")
    cols = {"n": g.size()}
    for m in METRICS:
        cols.update({f"{m}_median": g[m].median(), f"{m}_q25": g[m].quantile(0.25),
                     f"{m}_q75": g[m].quantile(0.75)})
    return pd.DataFrame(cols)


def gage_stats(y):
    """Median yearly NSE per held-out inflow gage (pooled over basins it is an inflow to)."""
    return (y.groupby(["inflow_gage", "inflow_name"])["nse"].agg(nse_median="median", n_years="size")
            .reset_index().sort_values("nse_median", ascending=False))


def plot_map(ax, gstats):
    """Inflow gages colored by median NSE over the UCRB, with the same simple layers as
    plots/bigmap.jpg (scripts/v2/example_basin.py): basin outline, reservoirs > 0.5 km2,
    Esri terrain + USGS hydrography tiles, north arrow, 100 km scale bar."""
    import geopandas as gpd
    import matplotlib.patches as mpatches
    import numpy as np

    crs = "EPSG:3857"
    ucol = gpd.read_parquet(C.UCOL_PATH).to_crs(crs)
    res = gpd.read_parquet(C.RESERVOIRS_PATH)
    res = res[res["areasqkm"] > 0.5].to_crs(crs)
    res.geometry = res.geometry.make_valid()
    gages = gpd.read_parquet(C.GAGES_PATH)
    gages.index = gages.index.map(C.fix_gage_id)
    pts = gages[["geometry"]].join(gstats.set_index("inflow_gage"), how="inner").to_crs(crs)

    ucol.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2, zorder=3)
    res.plot(ax=ax, edgecolor="cyan", facecolor="lightblue", linewidth=0.5, alpha=0.5, zorder=2)
    pts = pts.sort_values("nse_median")
    sc = ax.scatter(pts.geometry.x, pts.geometry.y, c=pts["nse_median"], cmap="viridis",
                    vmin=0, vmax=1, s=60, edgecolor="black", linewidth=0.5, zorder=5)
    try:
        import contextily as cx
        cx.add_basemap(ax, source=cx.providers.Esri.WorldTerrain, attribution=False)
        cx.add_basemap(ax, source="https://basemap.nationalmap.gov/arcgis/rest/services/"
                                  "USGSHydroCached/MapServer/tile/{z}/{y}/{x}", attribution=False)
    except Exception as e:   # no contextily or no internet: plot without the basemap
        print(f"basemap skipped: {e}")

    ax.annotate("N", xy=(0.06, 0.95), xytext=(0.06, 0.88), xycoords="axes fraction",
                arrowprops=dict(facecolor="black", width=3, headwidth=9), ha="center", va="center",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="black", alpha=0.8))
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lat = np.degrees(2 * np.arctan(np.exp((y0 + y1) / 2 / 6378137)) - np.pi / 2)
    length = 100_000 / np.cos(np.radians(lat))   # 100 km on the ground in Web Mercator metres
    sx, sy = x0 + (x1 - x0) * 0.05, y0 + (y1 - y0) * 0.04
    ax.plot([sx, sx + length], [sy, sy], color="black", linewidth=3, zorder=6)
    ax.text(sx + length / 2, sy + (y1 - y0) * 0.012, "100 km", ha="center", va="bottom",
            fontsize=10, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))
    ax.set_axis_off()
    ax.legend(handles=[mpatches.Patch(facecolor="none", edgecolor="black", linewidth=2,
                                      label="Upper Colorado Watershed"),
                       mpatches.Patch(facecolor="lightblue", edgecolor="cyan", label="Major Reservoir")],
              loc="lower right", framealpha=0.9)
    return sc


def plot_by_year(stats, gstats, out):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
    for ax, (m, label) in zip(axes, METRICS.items()):
        ax.fill_between(stats.index, stats[f"{m}_q25"], stats[f"{m}_q75"], alpha=0.3,
                        label="interquartile range")
        ax.plot(stats.index, stats[f"{m}_median"], label="median")
        if m == "pbias":
            ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("Water year")
        ax.set_ylabel(label)
        ax.legend()

    ax = fig.add_subplot(gs[:, 1])
    sc = plot_map(ax, gstats)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Median yearly NSE of held-out inflow gage")

    fig.suptitle("Ungaged Inflows Leave-One-Out Validation")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hydrographs(y, out):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out) as pdf:
        for gage, grp in y.groupby("gage", sort=True):
            df = pd.read_parquet(C.NATURALIZED_FULL_DIR / f"{gage}.parquet").set_index("date")
            for ig, r in grp.groupby("inflow_gage"):
                cols = [f"igage_{ig}", f"test_{ig}"]
                if not set(cols) <= set(df.columns):
                    continue
                d = df[cols].dropna(how="all")
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(d.index, d[cols[0]], label="observed (naturalized)")
                ax.plot(d.index, d[cols[1]], label="predicted from other inflows")
                ax.set_ylabel("cfs")
                ax.legend(loc="upper left")
                ax.set_title(f"{r['inflow_name'].iloc[0]} ({ig}) held out at {r['name'].iloc[0]} "
                             f"({gage})\nmedian yearly NSE {r['nse'].median():.2f} over "
                             f"{len(r)} scored years (all predicted years shown)", loc="left")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hydrographs", action="store_true",
                   help="also write a PDF of daily observed vs predicted for every inflow gage")
    args = p.parse_args()

    y = load()
    if y.empty:
        raise SystemExit("no gage-year is within the share bounds")
    stats = yearly_stats(y)
    gstats = gage_stats(y)

    d = C.VALIDATION_PLOT_DIR
    d.mkdir(parents=True, exist_ok=True)
    stats.to_csv(d / "inflow_validation_by_year_stats.csv")
    gstats.to_csv(d / "inflow_validation_by_gage_stats.csv", index=False)
    plot_by_year(stats, gstats, d / "inflow_validation_by_year.png")
    if args.hydrographs:
        plot_hydrographs(y, d / "inflow_validation_hydrographs.pdf")

    print(f"{len(y)} gage-years ({y['inflow_gage'].nunique()} inflow gages, {y['gage'].nunique()} basins), "
          f"water years {stats.index.min()}-{stats.index.max()}")
    print(stats.round(2).to_string())
    print(f"plots in {d}")


if __name__ == "__main__":
    main()
