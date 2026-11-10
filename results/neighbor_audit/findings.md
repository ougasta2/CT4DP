# Connected-area candidate audit (2024)

## Recommendation

1. **DK1–DE-LU:** strongest first implementation candidate. Direct connection,
   substantial wind and solar, hourly price and country-level German generation/
   load at 15-minute resolution are available. German national generation is a
   proxy for the DE-LU bidding zone, not northern Germany alone or necessarily
   every component of DE-LU. 2024 provides only 82 discordant negative hours,
   all on the German side, concentrated into an unknown number of independent
   events; it is a limited sample for rare-event classification.
2. **DK1–NL:** useful second pair connected by COBRAcable. Wind and solar are
   material, the national granularity is broadly aligned with the NL zone, and
   price, wind, solar and load fields are present. Fossil gas is also important;
   similarity to Denmark is partial. There are 151 discordant negative hours
   in 2024, not independent events.
3. **DK2–SE4:** best contrast pair and promising for generalization, rather than
   a claim that its system closely matches Denmark. 2024 has 383 discordant
   negative hours, mostly SE4-only. Swedish national production is not an SE4
   feature. eSett's area-specific production and consumption API returned
   complete non-null wind, solar, production total and consumption total in
   each of two one-week samples. Check year-round coverage, measurement and
   redistribution terms before training.
4. **DK1–NO2:** useful hydro-dominated contrast, not a similar-mix comparison.
   eSett also supplies plausible-looking area-specific sample records and
   Energy-Charts supplies NO2 hourly prices, but Norwegian country-wide solar
   was missing in 3,047 of 8,784 records. 2024 has 169 discordant negative hours.
5. **DK1–SE3:** physical connection and eSett sample data exist, but the sampled
   Energy-Charts SE3 price license expressly prohibits external use of raw or
   derived data. Find an independently authorized price source before use in a
   shared report or repository. No SE3 price payload is saved here.

Availability does not establish model transfer, equal generation accounting,
causal effects, or comparable event dynamics. Pair rankings are a practical
screening judgment, not a quantitative distance between generation mixes.

## Actual checks

Energy-Charts v2 was queried on 2026-11-10 for the full local calendar year 2024.
The seven price-zone responses had 8,784/8,784 timestamps, no duplicate keys,
and non-null hourly prices. Only price payloads whose own license field says
CC BY 4.0 are retained, with attribution in the manifest. SE3's restricted
response and price-derived statistics are not retained. The DK1/DK2 prices
were independently joined by UTC hour to this repository's Energinet data:
8,784 rows each, zero disagreements on the negative-price indicator, maximum
absolute difference EUR 0.000029/MWh, consistent with precision differences.

Germany and Netherlands public net generation endpoints each returned 35,136
quarter-hour timestamps, all non-null for wind onshore, wind offshore, solar
and load; Denmark, Sweden and Norway country endpoints returned hourly data.
Country data and settlement data have different definitions: verify coverage,
load accounting, self-consumption and units before deriving comparable ratios.
For quarter-hour MW rates, take a four-quarter time-weighted mean to get average
hourly MW (or integrate rates to MWh); do not sum MW values as hourly energy.
Complete keys and non-null values do not establish accuracy or measurement
equivalence, and a 2024 check does not validate a longer historical range.

eSett `EXP16/Volumes` and `EXP15/Consumption` were called separately for SE3,
SE4 and NO2, for 1–7 January and 1–7 July 2024 UTC. Each area/window/endpoint
returned 672 of 672 expected quarter-hour timestamps, with non-null wind, solar
and production total, or consumption total, respectively. The API describes
values as MWh and returns negative consumption; sampling alone does not establish
unit conversion, missing-asset coverage or historical consistency. Multiple
MBA parameters are aggregated by the service, so query one area at a time.
The sample records were not saved because redistribution terms remain unverified.

Hourly price pair counts use strict `< 0`, UTC alignment and are descriptive.
Event counts do not establish independence or significance. For each pair,
`a_only + b_only + both = any_negative`, and discordant is `a_only + b_only`.
The 2024 results were inspected during selection and cannot serve as an untouched
confirmatory test. File `paired_events_2024.csv` retains all denominators.

## Inputs and sources

- [Energy-Charts API](https://api.energy-charts.info/) provides the audited
  responses; `download_manifest.json` records exact requests, license strings
  and SHA-256 hashes. Attribution: Energy-Charts / Fraunhofer ISE; price source
  Bundesnetzagentur | SMARD.de, CC BY 4.0 as stated in each response.
- [eSett Open Data API](https://api.opendata.esett.com/) documents the
  area-level endpoints. Dataset use and redistribution terms still need review.
- [Energinet's 2025 market report](https://en.energinet.dk/media/v2fhaaxv/250226-electricity-market-report-2025.pdf)
  discusses Denmark's connected markets and structural changes.
- [IEA Denmark report](https://www.oecd.org/content/dam/oecd/en/publications/reports/2023/12/denmark-2023_7d3ba636/755105d6-en.pdf)
  lists interconnections; [Fraunhofer ISE](https://www.ise.fraunhofer.de/en/press-media/press-releases/2025/public-electricity-generation-2024-renewable-energies-cover-more-than-60-percent-of-german-electricity-consumption-for-the-first-time.html)
  reports Germany's wind generation; [Statistics Netherlands](https://www.cbs.nl/item?sc_itemid=50c41347-bb40-4613-a10e-ac27dec5bd79&sc_lang=en-gb)
  reports Dutch wind, solar and gas generation; [Sweden's regulator](https://ei.se/download/18.6fe475b619b356781735d3/1766159092506/Swedens-electricity-and-natural-gas-market-2024-Ei-R2025-19.pdf)
  reports Sweden's national mix, not SE4's mix; [Statnett](https://www.statnett.no/en/for-stakeholders-in-the-power-industry/data-from-the-power-system/)
  describes Norway's hydropower-dominated system.

Reproduce the offline pair checks with `.venv/Scripts/python scripts/summarize_neighbor_prices.py`.
Reacquiring network samples requires `.venv/Scripts/python scripts/audit_neighbor_sources.py`
and `.venv/Scripts/python scripts/audit_esett_sample.py` with network access.
