"""
Compare estimated natural flow (Q_NAT2) with USBR natural flow at the gages USBR covers, monthly.

Standalone port of the "Comparison to USBR Accounting" block in scripts/v2/workflow.py
(~lines 1885-1990), reading the current pipeline output instead of the old CSVs.

    python scripts/v3/compare_usbr.py
    python scripts/v3/compare_usbr.py --out some/other.png

For each gage with USBR data: days where both Q_NAT2_cfs and Q_NATUSBR_cfs exist are averaged to
monthly (USBR is monthly spread to daily, so its monthly mean is the USBR value). Metrics drop
months where either value is <= 0, as in workflow.py.

The right panel replaces workflow.py's NSE bars with KGE (Gupta et al., 2009) and its components:
    KGE = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (alpha - 1)^2)
    r     Pearson correlation
    beta  bias ratio        mean(sim) / mean(obs)
    alpha variability ratio std(sim) / std(obs)
Every term is 1 for a perfect match. obs = USBR, sim = Q_NAT2. The figure shows r and KGE as bars
from 0, and beta and alpha as % departures from 1 (0 = perfect); rivers are ordered by mean flow,
smallest at the top.

Reads  shiny-app/ucol_natural/timeseries/<gage>.parquet and basin_info.csv (names)
Writes plots/usbr_comparison.png            log-log scatter (all months) + KGE components by river
       plots/usbr_comparison_by_gage.csv     R2, NSE, PBIAS, KGE, r, beta, alpha, months per gage
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[2]
TS_DIR = REPO / "shiny-app" / "ucol_natural" / "timeseries"
OUT_DIR = REPO / "plots"

# right-panel columns: (metric column, axis title, perfect value). Bias and variability are shown
# as % departures from 1, so 0 is a perfect match and the bar points over (+) or under (-).
KGE_PARTS = [
    ("r", "Correlation\n$r$", 1),
    ("bias_pct", "Bias (%)\n$100\\,(\\mu_{sim}/\\mu_{obs} - 1)$", 0),
    ("var_pct", "Variability (%)\n$100\\,(\\sigma_{sim}/\\sigma_{obs} - 1)$", 0),
    ("KGE", "KGE", 1),
]


def _valid(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim) & (obs > 0) & (sim > 0)
    return obs[mask], sim[mask]


def calc_metrics(obs, sim):
    """R^2, Nash-Sutcliffe efficiency (NSE) and percent bias (PBIAS), months with both > 0."""
    obs, sim = _valid(obs, sim)
    r2 = pearsonr(obs, sim)[0] ** 2 if len(obs) > 1 else np.nan
    nse = 1 - (np.sum((sim - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2))
    pbias = 100 * np.sum(sim - obs) / np.sum(obs)
    return r2, nse, pbias


def calc_kge(obs, sim):
    """KGE, r, beta (bias ratio), alpha (variability ratio); same months as calc_metrics."""
    obs, sim = _valid(obs, sim)
    r = pearsonr(obs, sim)[0] if len(obs) > 1 else np.nan
    beta = sim.mean() / obs.mean()
    alpha = sim.std() / obs.std()
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (alpha - 1) ** 2)
    return kge, r, beta, alpha


def load_monthly():
    """Monthly Q_NAT2 and USBR for every gage with USBR data, stacked with a 'gage' column."""
    dfs = []
    for path in sorted(TS_DIR.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["date", "Q_NAT2_cfs", "Q_NATUSBR_cfs"])
        if not df["Q_NATUSBR_cfs"].notna().any():
            continue
        df = df.set_index(pd.to_datetime(df["date"]))
        mdf = df[["Q_NAT2_cfs", "Q_NATUSBR_cfs"]].dropna().resample("MS").mean()
        mdf["gage"] = path.stem
        dfs.append(mdf)
    return pd.concat(dfs)


def plot_scatter(ax, x_obs, y_sim, stats):
    ax.scatter(x_obs, y_sim, color="#1f77b4", alpha=0.6, edgecolors="none", s=25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    min_val = min(x_obs[x_obs > 0].min(), y_sim[y_sim > 0].min())
    max_val = max(x_obs.max(), y_sim.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="1:1 Line")
    ax.set_xlim(100, max_val)
    ax.set_ylim(100, max_val)
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="gray"))
    ax.set_xlabel("USBR Natural Streamflow (cfs)", fontsize=11)
    ax.set_ylabel("Natural Streamflow (cfs)", fontsize=11)
    ax.set_title("Comparison to USBR Accounting", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)


def plot_kge_parts(fig, spec, df_metrics):
    """One narrow bar column per KGE component plus KGE, rivers on a shared y axis (row order of
    df_metrics, first row at the bottom). r and KGE run from 0 to 1; bias and variability are %
    departures from a perfect match, with a dotted line at 0."""
    inner = gridspec.GridSpecFromSubplotSpec(1, len(KGE_PARTS), subplot_spec=spec, wspace=0.22)
    y_pos = np.arange(len(df_metrics))
    axes = []
    for j, (col, title, perfect) in enumerate(KGE_PARTS):
        ax = fig.add_subplot(inner[j])
        axes.append(ax)
        vals = df_metrics[col].values
        is_kge = col == "KGE"
        ax.barh(y_pos, vals, height=0.65, color="#1d4e89" if is_kge else "#5b8cc0")
        if perfect == 0:
            ax.axvline(0, color="black", lw=0.8, ls=":")
            lim = 1.15 * np.abs(vals).max()
            ax.set_xlim(-lim, lim)
        else:
            ax.set_xlim(0, 1)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(df_metrics) - 0.4)
        ax.set_yticks(y_pos)
        if j == 0:
            ax.set_yticklabels(df_metrics["river_name"], fontsize=10)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=11, fontweight="bold" if is_kge else "normal")
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
        ax.tick_params(axis="x", labelsize=9)
    return axes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT_DIR / "usbr_comparison.png")
    args = p.parse_args()

    names = (pd.read_csv(TS_DIR / "basin_info.csv", dtype={"gage": str})
             .set_index("gage")["name"].to_dict())
    bigdf = load_monthly()

    df_clean = bigdf[["Q_NATUSBR_cfs", "Q_NAT2_cfs", "gage"]].dropna()
    x_obs = df_clean["Q_NATUSBR_cfs"].values
    y_sim = df_clean["Q_NAT2_cfs"].values
    r2_all, nse_all, pbias_all = calc_metrics(x_obs, y_sim)
    kge_all = calc_kge(x_obs, y_sim)[0]

    rows = []
    for gage, group in df_clean.groupby("gage"):
        o, s = group["Q_NATUSBR_cfs"].values, group["Q_NAT2_cfs"].values
        r2, nse, pbias = calc_metrics(o, s)
        kge, r, beta, alpha = calc_kge(o, s)
        rows.append({"gage": gage, "river_name": names.get(gage, gage), "n_months": len(group),
                     "mean_usbr_cfs": o.mean(), "R2": r2, "NSE": nse, "PBIAS": pbias, "KGE": kge,
                     "r": r, "beta": beta, "alpha": alpha,
                     "bias_pct": 100 * (beta - 1), "var_pct": 100 * (alpha - 1)})
    # sorted by size (mean USBR flow): largest first = bottom row, smallest river at the top
    df_metrics = (pd.DataFrame(rows).sort_values("mean_usbr_cfs", ascending=False)
                  .reset_index(drop=True))

    fig = plt.figure(figsize=(17, 6.5))
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3], wspace=0.5)
    stats = (f"$R^2$ = {r2_all:.3f}\nNSE = {nse_all:.3f}\nKGE = {kge_all:.3f}\n"
             f"PBIAS = {pbias_all:.1f}%")
    plot_scatter(fig.add_subplot(outer[0]), x_obs, y_sim, stats)
    kaxes = plot_kge_parts(fig, outer[1], df_metrics)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.1)
    # panel title centred over the four component columns
    x0, x1 = kaxes[0].get_position().x0, kaxes[-1].get_position().x1
    fig.text((x0 + x1) / 2, 0.955, "KGE and its Components by River", ha="center",
             fontsize=12, fontweight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    df_metrics.sort_values("mean_usbr_cfs").to_csv(OUT_DIR / "usbr_comparison_by_gage.csv", index=False)

    print(f"{len(df_metrics)} gages, {len(df_clean)} gage-months: R2 {r2_all:.3f}, "
          f"NSE {nse_all:.3f}, KGE {kge_all:.3f}, PBIAS {pbias_all:.1f}%")
    print(df_metrics.sort_values("mean_usbr_cfs").round(3).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
