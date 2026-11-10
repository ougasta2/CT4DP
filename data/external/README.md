# Archived weather forecasts and Brent inputs

`weather/*_jma_day2.json.gz` holds Open-Meteo JMA GSM archived `previous_day2`
temperature, 10 m wind speed and cloud cover, with the exact URL, retrieval time
and SHA-256 of the HTTP response. Weather data by [Open-Meteo](https://open-meteo.com/)
using JMA forecasts; API data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
according to [Open-Meteo's licence page](https://open-meteo.com/en/licence).
The public endpoint is used
for this noncommercial research. Locations and model specification are in
`scripts/fetch_forecast_inputs.py`.

`fred/brent.csv.gz` contains EIA's Brent spot-price series through FRED. FRED
marks the series copyrighted; this local cache and fitted models/predictions
that use it are Git-ignored. Both the forecast run publication timestamps and
historical oil release/vintage availability remain unverified, despite the
conservative 48-hour weather lead and 14-day oil observation lag.

`scripts/fetch_forecast_inputs.py` fetches missing sources. The forecast study
also requires the prepared multi-area price files; `data/multiarea/README.md`
explains the local-only SE4 input and source-specific preparation.

`gas/gasradar_daily.csv.gz` stores the Gasradar.nl gas-day price proxy distributed
from EnergyZero, with URL, retrieval timestamp and HTTP response hash. Gasradar
states CC BY 4.0 for its compilation and requests attribution to Gasradar.nl and
EnergyZero. It is a TTF-like indicator, not a verified exchange TTF settlement
feed. Historic publication timestamps and revisions are not captured.

`energinet_forecasts/` contains original API response envelopes for Denmark's
2020–2024 hourly wind and solar forecasts. Source: Energinet,
[Energi Data Service](https://www.energidataservice.dk/),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The separately
reported hindsight experiment filters and aggregates these original data;
the `ForecastDayAhead` values are published at 18:00, after the 10:00 cutoff.
