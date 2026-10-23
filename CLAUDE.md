# CLAUDE.md

## Project goal
Build a dataset of **daily natural (naturalized) streamflow estimates for USGS gages in the
Upper Colorado River Basin (UCRB / "UCOL")**, plus a Shiny app that lets hydrologists and
non-specialists view and export that data. The app and dataset will accompany a journal
publication.

Not everything here is automated, and it shouldn't all be. One-off steps (basin delineation,
gage selection, irrigation mapping, distance matrix, reservoir lists) were run once and their
outputs are reused as static inputs. Only data that changes over time is automated.

## Where the work lives
- **`shiny-app/ucol_natural/`**: the app and its data pipeline. **New work goes here.**
  - `app.py`: Shiny for Python (Express) app. ipyleaflet map (UCOL, gages, basins, diversions,
    reservoirs) and a Plotly hydrograph: observed (`Q_unfilled_*`), estimated natural
    (`Q_NAT2_*`), USBR natural (`Q_NATUSBR_*`), stacked consumptive use; CSV download.
    Paths are relative to `app.py` (`APP_DIR`). It reads `timeseries/<gage>.parquet`
    and falls back to the old `<gage>.csv` when no parquet exists.
  - `pipeline/`: data pipeline (see below).
  - `spatial_data/`: static inputs (basins, gages, reservoirs, DRI evap reservoir points,
    diversion points, `basin_irrigation_percentages.csv`, `dist_matrix.csv`,
    `usbr_nat_flows.csv`).
  - `timeseries/`: per-gage output read by the app (currently old CSVs; parquets once the
    pipeline has run).
- **`scripts/v2/workflow.py`**: the original interactive workflow (~2,700 lines, run block by
  block). Its naturalization loop has been ported to `pipeline/naturalize.py`; keep
  `workflow.py` as the record of one-off steps (gage search, delineation, attributes) and
  don't refactor it.
- **`scripts/v3/ucrb_diversions.py`**: the USGS diversion download script (Knight & Lopez, USGS),
  adapted into `pipeline/diversions.py`.
- **Out of scope (LSTM project):** `scripts/v1/`, the basin-selection / PREP NH sections at the
  end of `workflow.py`, `scripts/v2/eval.py`. Don't change or build on these unless asked.

## Data pipeline (`shiny-app/ucol_natural/pipeline/`)
```
pip install -r shiny-app/ucol_natural/pipeline/requirements.txt
python shiny-app/ucol_natural/pipeline/update_inputs.py   # daily: download/refresh inputs
python shiny-app/ucol_natural/pipeline/naturalize.py      # build timeseries/<gage>.parquet
```
- `common.py`: all paths and shared helpers (`fix_gage_id`, `fill_discharge_data`,
  `merge_window`). Paths are relative to the file, so scripts run from any folder.
- `update_inputs.py`: USGS streamflow (pygeohydro NWIS, falling back to dataretrieval
  waterdata), diversions (`diversions.py`), DRI reservoir evap. Each is cached in
  `pipeline/data/` (gitignored). After the first full run, only a recent window is
  re-downloaded and merged in (streamflow 120 d, diversions 730 d, evap 365 d). If a site
  fails, its cached copy is used. Flags: `--only`, `--full`, `--plots`,
  `--refresh-evap-reservoirs`.
- `diversions.py`: per-source fetchers plus the USGS filter (`divfilter`) and gap fill
  (`fill_missing_diversion_records`). It is tested against the original USGS functions and
  matches to rounding error.
- `naturalize.py`: port of workflow.py's "MERGE DIVERSIONS WITH STREAMFLOW" loop. Basins are
  processed smallest to largest; final `Q_NAT2_cfs` is kept in memory (`nat_store`) for
  below-reservoir basins. Outputs: `timeseries/<gage>.parquet` (app columns),
  `pipeline/data/naturalized_full/`, `pipeline/data/nat_results.csv`,
  `spatial_data/ucrb_diversion_master_table.parquet`. `--gages` runs a subset.
  Below-reservoir validation: each inflow gage is held out and predicted from the others
  (same dates as `frac_in`, per-year inflow set and `frac_covered`, skipped in years it carries
  < 5% of the basin's estimated natural flow, `MIN_VALIDATION_SHARE`). NSE/PBIAS/r go to
  `pipeline/data/inflow_validation.csv` (per basin + inflow gage) and
  `inflow_validation_by_year.csv`. Whole-record PBIAS is ~0 by construction (in-sample ratio).
- `plot_inflow_validation.py`: plots those CSVs to `pipeline/data/plots/` (`--hydrographs`
  adds a daily observed-vs-predicted PDF).
- `download_nhd.py` (one-off, run by Will): NHD HR flowlines for UCRB HU4s 1401-1408 from the
  USGS staged HU4 products (GeoPackage, else Shapefile; NHD is retired since 2023-10 but still
  hosted). Keeps every flowline type, labelled `flow_class` (stream_perennial / intermittent /
  ephemeral, canal_ditch, artificial_path, ...), EPSG:5070, 2-D. Writes to
  `pipeline/data/nhd/` (gitignored, large): per-HU4 and combined UCRB geoparquet,
  `basins/<gage>.parquet` clipped to each whole drainage, `basin_flowline_km.csv`.
  Artificial paths include wide mainstem river centerlines, so keep them with perennial streams.
- `dist2stream.py` (one-off, Google Earth Engine, run by Will; login details at the top):
  effective distance sqrt(mean d^2) from IrrMapper v1.2 irrigated pixels (1986-2024) to NHD
  flowlines, per basin (whole drainage) and year; unmapped years get the basin median. Default
  stream set `--classes stream_perennial canal_ditch artificial_path` (tag `art-can-per`); each
  class is uploaded to EE once, so other sets only upload new classes. `submit` / `status` /
  `collect` -> `spatial_data/basin_dist2stream_<tag>.csv` (+ `_by_year`), plots in
  `pipeline/data/plots/`. Replaced `irrigation_distance.py` (WWF rivers). Not yet used by
  naturalize.py (Will reviews results first; T and S come later).
  Tested on gages 09149500 and 09251000 (inputs rebuilt from old output):
  Q_NAT / Q_NAT2 / Q_NATUSBR match to rounding error.
- **Required inputs not in the repo yet:** `pipeline/inputs/diversions/` needs
  `ucrb_diversion_master_table.csv` and the `nmose_`, `utdwr_`, `cuwcd_` and
  `wyseo_historical_data/` folders, copied from
  `N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\diversion\input`.
- **API keys come from environment variables:** `API_USGS_PAT`, `CDSS_API_KEY`,
  `DRI_EVAP_API_KEY`. Never hard-code keys.

## Status (end of 2026-10-22 session)
Done on `cc`: moved v1 folders into `scripts/v1/`; added this file; built the pipeline; switched
the app to parquet and relative paths. 2026-10-22: fixed the below-reservoir leave-one-out
validation, added the 5% share rule, validation CSVs and `plot_inflow_validation.py`
(stub-tested only; Q_NAT2/Q_NAT3 unchanged). Open question: make the validation out-of-sample
(fit the ratio without the scored year).
2026-10-23: added `download_nhd.py` (Will has run it for Lee's Ferry) and `dist2stream.py`
(stub-tested with a fake `ee`; not yet run).

**Next steps**
1. Will: copy the diversion inputs, set the API keys, and run `update_inputs.py` then
   `naturalize.py` locally. Fix whatever breaks: the downloads, spatial joins, parquet writing
   and the below-reservoir path haven't been run yet. Then compare the new parquets with the
   old CSVs and delete the CSVs.
2. Decide how to treat recent diversions. The USGS fill sets the end of a partly reported
   year to 0 (a fully missing year gets monthly means), so late-reporting sites show zero
   diversions for recent weeks, and natural flow equals observed flow there. The app should
   at least flag recent values as provisional.
3. Fix `shiny-app/ucol_natural/requirements.txt` (still the Shiny template; `shared.py` is
   a template leftover too).
4. Slim the timeseries data (drop columns the app never uses).
5. Add a nightly GitHub Action: keys in repo Secrets, outputs published to a Release asset or
   cloud storage, **not committed to git** (the repo history is already ~3 GB).
6. Deployment plan: archive a frozen dataset plus code on Zenodo/HydroShare (DOI) for the
   paper; host the app on Posit Connect Cloud.

**Known issues, kept as-is because changing them changes results (ask before changing):**
- Muskingum routing calculates velocity from mean flow in cfs, while the formula
  (`v = a*Q^b`) expects cms.
- As in the USGS script, NMOSE sites are read from raw downloads, so their historical records
  are not merged in.
- Exposed API keys: a USGS key is in `scripts/v2/flow_routing.py` and
  `calibrate_return_flow.py`, and a DRI key is in `scripts/v2/evap.py`. They need rotating
  and removing.

## Data and paths
- Old scripts (`scripts/v2`) are Windows-first, run from the repo root, and read from
  `N:\Research\Kampf\...` (lab drive) and `G:\My Drive\natural_streamflow_colab` (LSTM).
  Don't "fix" those paths. New code uses `pathlib` relative to the file.
- Gage IDs are strings and must keep leading zeros (e.g. `'09010500'`); use `fix_gage_id`.
- Projected CRS for areas and spatial joins: EPSG:5070. The app uses EPSG:4326; the USGS
  diversion tables use 26912.
- Water year = Oct 1 – Sep 30. Data start 1979-10-01; output starts 1981.

## Timeseries columns
Unit suffix `_cms` / `_cfs` / `_mmd`. Key columns: `Q_cfs` (observed, gap-filled),
`Q_unfilled_cfs` (observed), `Q_CU_cfs` (total consumptive use), `<type>_cfs` per CU type
(irrigation, municipal, interbasin, industrial, hydropower, intrabasin, transbasin, evap),
`Q_NAT_cfs`, **`Q_NAT2_cfs` (main product)**, `Q_NAT3_cfs` (routed, reservoir basins),
`Q_NATUSBR_cfs`. The app depends on these names, so a rename must update `naturalize.py`
and `app.py`.

## Environment
- Will uses Windows, conda env `gis_env`, PyCharm.
- Cloud Claude sessions may be unable to install packages (PyPI blocked), so test
  non-spatial logic with stubs and leave full runs to Will.

## Working rules
- Make focused, minimal edits. Port logic faithfully, and flag any scientific changes
  rather than making them silently.
- Don't run full downloads or the full naturalization without asking.
- Work on the `cc` branch; `main` is the pre-Claude baseline (it has one later commit,
  `go`, with post-loop analysis in workflow.py that isn't on `cc`).
- Commit as Will (`wkeenan21`), with the Claude co-author trailer.
