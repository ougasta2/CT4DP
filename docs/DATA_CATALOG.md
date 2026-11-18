# Data catalog and setup

The data is already organized by source and study. Existing paths are retained because the historical pipelines and notebooks use them directly. Do not move or rename caches without updating and validating every dependent script. Large raw files, prepared tables and outputs have different redistribution conditions.

| Location | Role | Availability / notes |
| --- | --- | --- |
| `data/decade/` and `results/decade/processed.csv.gz` | Danish historical inputs and processed DK1/DK2 prices used by the team runner | Bundled; see [decade inventory](../data/decade/README.md) and [protocol](DECADE_PROTOCOL.md). |
| `data/multiarea/energycharts/` | Neighbor-zone price histories and newer DK1/DK2 prices | Bundled source envelopes; see [multi-area inventory](../data/multiarea/README.md). Preserve the attribution in each source. |
| `data/external/weather/` | Archived JMA weather features for the five price zones | Bundled envelopes, including 2026 extensions; source and licensing in [external data](../data/external/README.md). Historical vintages are not verified. |
| `data/cache/`, `data/*.csv.gz`, `data/manifest.json` | Original two-year pilot responses, exports and checksums | Bundled; see [pilot inventory](../data/README.md). |
| `data/multiarea/esett/`, `data/external/fred/`, APG/EXAA and selected ENTSO-E backfills | Source-specific inputs for earlier experiments | Git-ignored pending redistribution review. They are **not** required by the new four-model runner, but some historical scripts require them. Obtain them through source-specific scripts where permitted. |
| `results/team/` | Locally generated predictions, timings and metrics | Git-ignored conservatively. Share only after confirming underlying data and derivative terms. |

The team runner calls the existing price loader and expects all five zones to cover the full period the loader constructs, including its 2026 extensions; it checks hourly continuity and fails on missing files or gaps. Weather loading also expects both historical and 2026 archives. Data preparation currently occurs in memory and is shared by all four models; it does not create a new normalized dataset on disk. `data/multiarea/paired_*.csv.gz` and eSett generation are not inputs to the new runner.

For the original pilot, consult [data/README.md](../data/README.md); for fuel, weather and archived forecasts, see [data/external/README.md](../data/external/README.md). Old historical scripts can depend on local-only files even if a saved aggregate table is committed. Source snapshots may contain revisions and retrieval timestamps are not proof of original publication time. Keep raw files, provenance and hashes intact and never commit credentials or unreviewed third-party data.
