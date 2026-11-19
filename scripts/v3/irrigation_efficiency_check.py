"""
Are the irrigation efficiencies above reservoirs plausible? A check using frac_in.

Below a reservoir, naturalize.py scales the gaged inflows up to the outlet:
    frac_in      = sum(inflow Q_NAT2) / sum(outlet Q_NAT)   over the inflow set's overlap days
    frac_covered = gaged inflow area / outlet area
Both natural flows add back irrigation consumption (efficiency x diversion), so they depend on the
irrigation efficiency (CU factor) used above the reservoir.

Plausible band: frac_covered <= frac_in <= 1.
  - frac_in > 1: the ungaged area would have to lose water on net. Possible (riparian ET,
    unaccounted CU), but with a large ungaged area it also fits efficiencies that are too low
    in the ungaged area (more consumed there, less reaching the reservoir).
  - frac_in < frac_covered: the ungaged land (mostly low and dry, near the reservoir) would yield
    more water per km2 than the gaged headwaters, which fits efficiencies that are too high
    (more water reaching the reservoir than we think).

Implied efficiency shift: with every basin's efficiency shifted by e (efficiency + e),
    frac_in(e) = (N_g + e * D_g) / (N_out + e * D_all)
N_g, N_out are the inflow and outlet natural-flow totals, D_g, D_all their irrigation diversion
totals (diversion = -irrigation_cfs / efficiency). Solving frac_in(e) = band edge gives the shift
that would bring the basin back into the band. If no efficiency in 0-1 does it, efficiency alone
can't explain the misfit. Approximations: one shift for all basins in the drainage, return-flow
lag ignored (it nets out over the multi-year overlap), nested below-reservoir inflows treated
like the others (their irrigation_cfs is still their whole-drainage diversion).

    python scripts/v3/irrigation_efficiency_check.py

Reads  shiny-app/ucol_natural/pipeline/data/frac_in.csv, timeseries/<gage>.parquet, basin_info.csv
Writes plots/frac_in_plausibility.png, plots/frac_in_plausibility.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
APP_DIR = REPO / "shiny-app" / "ucol_natural"
TS_DIR = APP_DIR / "timeseries"
FRAC_IN_PATH = APP_DIR / "pipeline" / "data" / "frac_in.csv"
OUT_DIR = REPO / "plots"
TOL = 0.02   # frac_in within this of a band edge counts as in band (fit noise)

_cache = {}


def series(gage):
    """Q_NAT2, Q_NAT and irrigation diversion (cfs) for one gage, cached."""
    if gage not in _cache:
        df = pd.read_parquet(TS_DIR / f"{gage}.parquet").set_index("date")
        cu = INFO.loc[gage, "irr_cu_factor"] if gage in INFO.index else np.nan
        irr = df["irrigation_cfs"] if "irrigation_cfs" in df.columns else pd.Series(0.0, df.index)
        div = (-irr / cu).fillna(0) if cu and np.isfinite(cu) else irr * 0
        _cache[gage] = pd.DataFrame({"nat2": df["Q_NAT2_cfs"], "nat": df["Q_NAT_cfs"], "div": div})
    return _cache[gage]


def totals(outlet, inflows, start, end):
    """N_g, N_out, D_g, D_all over the days where every inflow Q_NAT2 and the outlet Q_NAT exist."""
    nat = pd.concat([series(g)["nat2"].rename(g) for g in inflows]
                    + [series(outlet)["nat"].rename(outlet)], axis=1).loc[start:end].dropna()
    days = nat.index
    d_g = sum(series(g)["div"].reindex(days).fillna(0).sum() for g in inflows)
    d_all = series(outlet)["div"].reindex(days).fillna(0).sum()
    return nat[inflows].sum().sum(), nat[outlet].sum(), d_g, d_all, len(days)


def shift_to(target, n_g, n_out, d_g, d_all):
    """Efficiency shift e with (n_g + e d_g) / (n_out + e d_all) = target (nan if none)."""
    den = d_g - target * d_all
    return (target * n_out - n_g) / den if den != 0 else np.nan


def build_table():
    fr = pd.read_csv(FRAC_IN_PATH, dtype={"gage": str, "inflow_gages": str},
                     parse_dates=["overlap_start", "overlap_end"])
    # one row per basin and inflow set (frac_in is fitted once per set)
    sets = (fr.groupby(["gage", "inflow_gages"])
            .agg(name=("name", "first"), frac_in=("frac_in", "first"),
                 frac_covered=("frac_covered", "first"), n_years=("water_year", "size"),
                 start=("overlap_start", "first"), end=("overlap_end", "first"))
            .reset_index())
    rows = []
    for _, s in sets.iterrows():
        inflows = s["inflow_gages"].split(";")
        n_g, n_out, d_g, d_all, n_days = totals(s["gage"], inflows, s["start"], s["end"])
        lo, hi = s["frac_covered"], 1.0
        above, below = s["frac_in"] > hi + TOL, s["frac_in"] < lo - TOL
        target = hi if above else lo if below else np.nan
        e = shift_to(target, n_g, n_out, d_g, d_all) if np.isfinite(target) else 0.0
        eff = INFO.loc[s["gage"], "irr_cu_factor"]
        rows.append({**s.drop(["start", "end"]).to_dict(), "frac_in_check": n_g / n_out,
                     "ungaged_irr_share": (d_all - d_g) / n_out,  # ungaged diversion / outlet flow
                     "efficiency": eff, "shift_needed": e, "implied_efficiency": eff + e,
                     "status": "above 1" if above else "below area share" if below else "in band"})
    return pd.DataFrame(rows)


def plot(t, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.1]})

    # left: frac_in vs frac_covered, plausible band shaded
    x = np.linspace(0.3, 1.02, 50)
    ax1.fill_between(x, x, 1, color="#2e7d32", alpha=0.10, lw=0)
    ax1.plot(x, x, color="gray", lw=1, ls="--")
    ax1.axhline(1, color="gray", lw=1, ls="--")
    ax1.text(0.36, 0.385, "frac_in = area share", rotation=33, color="gray", fontsize=9)
    ax1.text(0.36, 1.005, "frac_in = 1", color="gray", fontsize=9, va="bottom")
    ax1.text(0.52, 0.80, "plausible:\nungaged land yields less\nper km² than headwaters",
             color="#2e7d32", fontsize=9)
    colors = {"in band": "#9e9e9e", "above 1": "#c62828", "below area share": "#1565c0"}
    for status, g in t.groupby("status"):
        ax1.scatter(g["frac_covered"], g["frac_in"], s=12 + 6 * g["n_years"], c=colors[status],
                    alpha=0.75, edgecolor="white", lw=0.5, label=status)
    for _, r in t[t["status"] != "in band"].sort_values("n_years").drop_duplicates("gage", keep="last").iterrows():
        ax1.annotate(r["gage"], (r["frac_covered"], r["frac_in"]), xytext=(4, 2),
                     textcoords="offset points", fontsize=7.5, color=colors[r["status"]])
    ax1.set_xlim(0.3, 1.02)
    ax1.set_xlabel("Gaged inflow area / outlet area (frac_covered)")
    ax1.set_ylabel("Gaged natural inflow / outlet natural flow (frac_in)")
    ax1.set_title("(a) Below-reservoir inflow sets (size = water years used)", loc="left", fontsize=11)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)

    # right: efficiency needed to bring each out-of-band basin back to the band edge
    o = (t[t["status"] != "in band"].sort_values("n_years")
         .drop_duplicates("gage", keep="last").sort_values("shift_needed").reset_index(drop=True))
    y = np.arange(len(o))
    for i, r in o.iterrows():
        c = colors[r["status"]]
        imp, cur = r["implied_efficiency"], r["efficiency"]
        ax2.plot(cur, i, "o", color="black", ms=5, zorder=3)
        if np.isfinite(imp) and 0 <= imp <= 1:
            ax2.annotate("", xy=(imp, i), xytext=(cur, i),
                         arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=14))
            ax2.text(imp + (0.02 if imp > cur else -0.02), i, f"{imp:.2f}", color=c, fontsize=8.5,
                     va="center", ha="left" if imp > cur else "right")
        else:
            ax2.text(cur + 0.03, i, "no efficiency in 0-1 explains it", color=c, fontsize=8.5,
                     va="center", style="italic")
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{r['name'][:38]} ({r['gage']})" for _, r in o.iterrows()], fontsize=8.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Irrigation efficiency (black: current; arrow: needed to reach the band edge)")
    ax2.set_title("(b) Efficiency shift that would explain the misfit", loc="left", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Plausibility of irrigation efficiencies above reservoirs", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")


INFO = pd.read_csv(TS_DIR / "basin_info.csv", dtype={"gage": str}).set_index("gage")


def main():
    t = build_table()
    OUT_DIR.mkdir(exist_ok=True)
    t.to_csv(OUT_DIR / "frac_in_plausibility.csv", index=False)
    print("max |frac_in - recomputed|:", (t["frac_in"] - t["frac_in_check"]).abs().max().round(4))
    print(t["status"].value_counts().to_string())
    cols = ["gage", "name", "n_years", "frac_covered", "frac_in", "ungaged_irr_share",
            "efficiency", "implied_efficiency", "status"]
    print(t[t["status"] != "in band"][cols].round(3).to_string(index=False))
    plot(t, OUT_DIR / "frac_in_plausibility.png")
    print(f"wrote {OUT_DIR / 'frac_in_plausibility.png'}")


if __name__ == "__main__":
    main()
