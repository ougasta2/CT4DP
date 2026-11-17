# DK1 benchmark data and attribution

The three compressed tables in `data/` are adapted from the following sources
under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

- Danish DK1 and DK2 spot prices: Energinet, [Energi Data Service](https://www.energidataservice.dk/), historical Elspotprices dataset.
- DE-LU, NL and SE4 day-ahead prices: Bundesnetzagentur | [SMARD.de](https://www.smard.de/) via the [Fraunhofer ISE Energy-Charts API](https://api.energy-charts.info/). The cached source responses identify Bundesnetzagentur | SMARD.de and CC BY 4.0.
- Weather data by [Open-Meteo.com](https://open-meteo.com/) using Japan Meteorological Agency GSM forecasts, as described on the [Open-Meteo licence page](https://open-meteo.com/en/licence).

The project aligned timestamps to Copenhagen delivery dates, selected complete
days, reshaped historical prices and archived weather into daily feature curves,
computed price lags and calendar features, averaged repeated clock hours and
interpolated the missing spring clock hour in daily curves. The hourly scoring
file retains the original hourly observations on the selected days. The
manifest specifies coverage and file hashes. The snapshot contains no eSett,
FRED, Gasradar, APG or EXAA inputs. The providers do not endorse this project.
