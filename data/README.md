# Shared study data

All inputs used for the DK1/DK2 calendar-year 2023/2024 pilot are included in this
repository. Clone it or use GitHub **Code → Download ZIP** to obtain them together
with the code. No API account or credentials are required.

| Location | Contents |
| --- | --- |
| `production_consumption_2023_2024.csv.gz` | 35,088 area-hour rows: all production/consumption input columns selected for this study |
| `elspotprices_2023_2024.csv.gz` | 35,088 area-hour rows: UTC time, bidding zone and EUR/MWh price |
| `cache/*.json` | Original API response envelopes, including metadata, URL, parameters and download timestamp; also includes the initial smoke samples |
| `manifest.json` | Source lineage, field names, row counts, file sizes and SHA-256 checksums for cache and raw exports |
| `../results/processed.csv.gz` | Full joined primary analysis table with engineered features, events and frozen-model cluster assignments |
| `../results/sensitivity/processed.csv.gz` | Full analysis table using estimated solar self-consumption in the solar feature |

The `.csv.gz` files are standard gzip-compressed CSVs. Pandas reads them directly:

```python
import pandas as pd
production = pd.read_csv('data/production_consumption_2023_2024.csv.gz')
prices = pd.read_csv('data/elspotprices_2023_2024.csv.gz')
analysis = pd.read_csv('results/processed.csv.gz')
production['HourUTC'] = pd.to_datetime(production['HourUTC'], utc=True)
```

These are the **complete study inputs**, not the entire upstream historical
database or every upstream column. Each source has 17,520 area-hours in 2023 and
17,568 in 2024, combining both price areas. Use `(HourUTC, PriceArea)` as the key.
Year membership follows Europe/Copenhagen; the first UTC hour is therefore
2022-12-31 23:00. The final boundary is exclusive. Do not sum overlapping cached
smoke and annual responses: use the two consolidated raw exports for full years.

For a full offline rerun after installing the README dependencies:

```powershell
.venv/Scripts/python scripts/run_pilot.py --offline
.venv/Scripts/python scripts/execute_notebook.py
```

`scripts/export_data.py` regenerates the raw exports and manifest entirely from
the bundled cache. The main analysis consumes those original cached responses;
the convenient raw exports need no manual editing. The original metadata and
historical snapshot are retained so later API revisions cannot silently change
the shared results.

Source: **Energinet, Energi Data Service**, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Original response snapshots retain source data; consolidated CSV exports and
processed tables are transformations by this project. No endorsement by Energinet
is implied. See [official terms](https://www.energidataservice.dk/terms-and-conditions),
[production metadata](https://api.energidataservice.dk/meta/dataset/ProductionConsumptionSettlement)
and [price metadata](https://api.energidataservice.dk/meta/dataset/Elspotprices).

Gross consumption is a settlement-derived proxy, and solar self-consumption is
estimated. Missing-value ambiguity and historical association limits continue to
apply; see the main README and pilot findings.
