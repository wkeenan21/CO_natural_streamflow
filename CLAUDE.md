# CLAUDE.md

## Project goal
Build a dataset of **daily natural (naturalized) streamflow estimates for USGS gages in the
Upper Colorado River Basin (UCRB / "UCOL")**, plus a Shiny app that lets hydrologists and
non-specialists view and export that data.

Not everything here is automated, and it shouldn't all be. Many steps were run once
(by hand, in an interactive session, or on another machine) and their outputs are reused.
Don't try to turn the whole project into a pipeline that runs end to end unless asked.

## Where the work lives
The project is mostly two files:

- **`scripts/v2/workflow.py`**: the naturalization workflow (~2,700 lines). It's a
  sequence of sections meant to be run interactively (cell by cell / block by block), not
  top to bottom. Main sections, in order:
  1. Find active USGS daily-value gages inside the UCOL boundary (gage 09380000, Lees Ferry),
     drop canals, tunnels, ditches and diversions; delineate basins (NLDI + manual DEM
     delineation); download streamflow into `big_flow.csv`. Wrapped in `skip = True`.
  2. Climate forcing: gridMET + SNODAS SWE extraction for each basin.
  3. **MERGE DIVERSIONS WITH STREAMFLOW**, the core naturalization loop (~lines 1280–1880).
     For each basin, smallest to largest:
     - `Q_NAT_cfs = Q_cfs − evap_cfs − Q_CU_cfs` (adds back consumptive use and reservoir evap).
       Consumptive-use multipliers by diversion type are in `useDict`; the irrigation
       multiplier is replaced per basin with that basin's irrigation efficiency.
     - Irrigation return flows are lagged with a Glover stream-depletion kernel →
       `Q_NAT2_cfs` (never allowed below 0).
     - Basins below major reservoirs are rebuilt from naturalized upstream inflow gages
       (non-overlapping sub-basins, scaled by `frac_in`), with Muskingum routing → `Q_NAT3_cfs`.
       Leave-one-out validation of the ungaged-inflow scaling is also done here.
     - Monthly NSE / PBIAS vs. USBR natural flows (`Q_NATUSBR_cfs`) are recorded in `results`.
     Writes one CSV per gage to `shiny-app/ucol_natural/timeseries/` (slim version) and
     to `N:\...\data\timeseries\selected_w_diversion\` (all diversion columns).
  4. Examine naturalization output (scatter plot + NSE by river).
  5. Basin attributes, reservoir storage, regulation scoring, nested-basin matrix, and
     train/test selection, then **PREP NH** (NeuralHydrology configs). Sections 5+ serve
     the LSTM project; see below.
- **`shiny-app/ucol_natural/app.py`**: Shiny for Python (Express syntax) app titled "Upper
  Colorado River Basin Naturalized Streamflow". Left: ipyleaflet map of UCOL, gages, basins,
  diversions and reservoirs, with gage search. Right: Plotly hydrograph of observed flow
  (`Q_unfilled_*`), estimated natural flow (`Q_NAT2_*`), USBR natural flow (`Q_NATUSBR_*`)
  and stacked consumptive-use components; unit selection (cms/cfs/mmd), date range and
  CSV download.

Other `scripts/v2/` files are supporting or one-off work: `evap.py` (reservoir
evaporation), `irrigation_data.py` (irrigation efficiency per basin), `merge_diversion.py`,
`flow_routing.py` and `calibrate_return_flow.py` (routing and return-flow experiments),
`swe.py`, and `example_basin.py` (figures for one basin). `eval.py` evaluates LSTM runs.
`select_basins.py` and `validate.py` are empty.

## Out of scope: LSTM modelling
`scripts/v1/` (configs, data_collection, modelling, plotting), the PREP NH and
train/test sections at the end of `workflow.py`, and `scripts/v2/eval.py` belong to a
related NeuralHydrology/LSTM project. **Don't change these, or build on them,
unless asked.** New work goes toward the naturalized dataset and the app.

## Data and paths
- Run scripts and the app **from the repo root**: paths are built from `os.getcwd()` and
  relative paths like `shiny-app\ucol_natural\...`.
- The code is Windows-first (raw-string backslash paths). Keep that style when editing
  existing code; prefer `os.path.join` / `pathlib` in new code.
- Many inputs live **outside the repo** and won't exist on other machines:
  - `ncwd = N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow` (lab drive: big_flow.csv,
    diversion records, evap, USBR natural flows, DEM/distance matrix, SNODAS)
  - `G:\My Drive\natural_streamflow_colab` (Google Drive, mostly LSTM)
  - occasional absolute `C:\Users\...` paths
  Don't "fix" or delete these paths. If a task needs one of these files, ask.
- App data (`shiny-app/ucol_natural/`): `spatial_data/*.parquet` (UCOL boundary, gages,
  basins, reservoirs, diversions), `attributes/`, and `timeseries/<gage>.csv` (~285 gages).
  These are **outputs of `workflow.py`**. Don't hand-edit or regenerate them unless asked.
- Gage IDs are strings and must keep leading zeros (e.g. `'09010500'`); use `fix_gage_id`.
- Projected CRS for area/spatial joins: EPSG:5070. The app uses EPSG:4326.
- Water year = Oct 1 – Sep 30. Main analysis period: WY1980–WY2025.

## Timeseries column conventions
Each unit suffix is `_cms`, `_cfs` or `_mmd`. Key columns: `Q_cfs` (observed, gap-filled),
`Q_unfilled_cfs` (observed, no fill), `Q_CU_cfs` (total consumptive use), per-type
`<type>_cfs` (irrigation, municipal, interbasin, industrial, hydropower, intrabasin,
transbasin, evap), `Q_NAT_cfs`, `Q_NAT2_cfs` (**the main natural-flow product**),
`Q_NAT3_cfs` (routed, reservoir basins), `Q_NATUSBR_cfs` (USBR monthly natural flows spread to daily).
The app depends on these names, so a rename must update both `workflow.py` and `app.py`.

## Environment
- Python, conda env `gis_env` (geopandas, shapely, rasterio, rasterstats, pygeohydro/pynhd,
  dataretrieval, pygridmet, rioxarray, pyarrow, scipy, matplotlib).
- App: `shiny`, `shinywidgets`, `ipyleaflet`, `ipywidgets`, `plotly`, `shinyswatch`,
  `geopandas`, `pandas`. Run from repo root: `shiny run shiny-app/ucol_natural/app.py`.
  (`shiny-app/ucol_natural/requirements.txt` and `shared.py` are leftovers from the
  Shiny template and don't reflect the app's real dependencies.)
- TODO (Will): Python version, and how/where the app is deployed.

## Working rules
- Make focused, minimal edits. `workflow.py` is long and interactive; don't refactor it
  wholesale, reorder its sections, or remove commented-out exploratory code unless asked.
- Don't run the full naturalization loop or anything that downloads large data without asking.
- Never commit credentials. Read API keys (e.g. `API_USGS_PAT`) from environment variables.
- Work happens on the `cc` branch; `main` is the pre-Claude baseline.
