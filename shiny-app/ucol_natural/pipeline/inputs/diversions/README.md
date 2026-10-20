# Diversion inputs

`diversions.py` needs these files from the original USGS diversion `input` folder
(`N:\Research\Kampf\Private\KeenanW\CO_natural_streamflow\data\diversion\input`):

- `ucrb_diversion_master_table.csv` — every diversion site, its data source and IDs
- `nmose_historical_data/` — manually collected New Mexico records
- `utdwr_historical_data/` and `cuwcd_historical_data/` — manually collected Utah records
- `wyseo_historical_data/` — Wyoming records, including sites with no online data

Do not put `api_token.txt` here; set the `CDSS_API_KEY` environment variable instead.
