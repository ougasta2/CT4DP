# Multi-area inputs: 2019–2025

This dataset supports exploratory pairs DK1–DE-LU, DK1–NL and DK2–SE4. It is
not a finalized forecasting dataset. The target under investigation is whether
exactly one area has a negative hourly price, **conditional on at least one area
being negative**. Both prices and realized energy are retrospective data.

## Period and sample size

The initial design uses 2019–2023 for training, 2024 for development, and 2025
for exploratory temporal evaluation. The earlier Danish study examined 2024 and
2025, and the multi-area 2024 price counts were already inspected when selecting
pairs; neither is a pristine confirmatory test. If a prospective test is needed,
freeze decisions first and obtain a genuinely unexamined future period.

| Pair | Training area-pair hours | Any negative hours | One negative hours | Local dates with one negative |
| --- | ---: | ---: | ---: | ---: |
| DK1–DE-LU | 43,824 | 1,078 | 408 | 89 |
| DK1–NL | 43,824 | 921 | 541 | 148 |
| DK2–SE4 | 43,824 | 608 | 353 | 72 |

Counts refer to 2019–2023 after the stated feature validity checks; a given
date can appear in multiple pairs, and hours within days are dependent. These
counts are adequate for an initial regularized, low-parameter feasibility
comparison, not proof of sufficient power, stable subgroup estimates, or enough
independent examples for flexible high-dimensional models. Leave-one-year-out
development tests and day-block uncertainty are needed; for example SE4's
2021 discordance occurred on four days. Per-year details are in
`results/multiarea/pair_coverage_events.csv`.

## Sources and access

- Danish settlement inputs and prices are the existing Energinet decade table;
  see `data/decade/README.md` and `docs/DECADE_PROTOCOL.md` for the 2020 solar
  repair, pre-2020 source-definition bridge and 2025 quarter-price conversion.
- `energycharts/` contains 35 gzip JSON response envelopes from the Energy-Charts
  v2 API: 2019–2025 prices for DE-LU, NL, SE4 and country-level public-net
  generation/load for DE and NL. Each source response declares CC BY 4.0; the
  price responses attribute Bundesnetzagentur | SMARD.de and the power responses
  energy-charts.info. Preserve the actual license and attribution in each payload.
  [API documentation](https://api.energy-charts.info/).
- `esett/` contains SE4 area-specific production and consumption from the eSett
  Open Data API, with an extra UTC boundary hour before 2019. Public access and
  download are described in [eSett's service terms](https://opendata.esett.com/terms),
  but explicit redistribution terms remain unresolved. These raw responses and
  the derived DK2–SE4 paired table are **local-only and Git-ignored**. Do not
  assume they will be present in a clone or publish them without resolving rights.
  Re-fetching requires network access. The `energycharts/` files and two other
  derived tables are not ignored by this rule.
- `results/multiarea/manifest.json` records request URLs, retrieval timestamps,
  hashes of original HTTP response bytes and local gzip envelopes. The response
  hash refers to the bytes before parsing, not to a reserialized JSON object.

## Preparation and limitations

Run from the project root:

```powershell
.venv/Scripts/python scripts/fetch_multiarea.py
.venv/Scripts/python scripts/prepare_multiarea.py
```

The fetcher caches successful responses; delete only a specifically identified
cache file if a deliberate refresh is required. The preparation script creates
one `paired_*.csv.gz` per pair and the coverage/event audits under
`results/multiarea/`. Each output row is an area-pair UTC hour. Calendar years
use Europe/Copenhagen boundaries; DST and leap days are retained. A source
quarter-hour block must contain all four aligned records; a single missing
quarter cannot silently pass as an hourly record. Missing inputs are not zeroed.

Price uses EUR/MWh; after the October 2025 day-ahead transition, four quarter
prices are averaged. Energy-Charts generation/load are MW rates and are averaged
over four quarters to hourly average MW; over a one-hour interval the numerical
value equals MWh. eSett production/consumption are reported as MWh per interval
and are summed over four quarters after the 22 May 2023 local settlement change;
hourly observations before the change are retained. eSett consumption is
reported negative and the prepared load field reverses its sign.

All 49 source-year combinations in `coverage_by_year.csv` have expected UTC
hours and complete required fields after aggregation, and all 61,368 hours per
pair pass finite/nonnegative generation and positive load checks. Completeness
does not prove that zero means measured zero, that all assets are included, or
that sources measure equivalent quantities. DE public-net generation is at
country scale while DE-LU is a price zone; NL is national; SE4 is a zone;
DK generation is a settlement proxy that includes estimated solar self-use.
Historical source definitions and the 2024 flow-based market change require
special consideration before fitting or interpreting cross-area models.

The eSett Open Data query for DK2 in 1–7 July 2024 returned null wind and solar
fields with production under `other`, so it cannot directly harmonize DK and SE4
generation categories; the DK side continues to use Energinet. Cross-source
solar/load consistency remains unresolved. Do not use absent components as zero
or infer causal effects from these settlement observations.
