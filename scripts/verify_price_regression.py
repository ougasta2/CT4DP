"""Rebuild inputs and verify all saved price predictions."""
import gzip
import io
import json
import joblib
import numpy as np
import pandas as pd
import train_forecast_models as prior
import train_latest_forecast as latest
import train_price_regression as reg


def main():
    raw=latest.price_pairs();weather={z:latest.weather_all(z) for z in reg.ZONES}
    with gzip.open('data/external/fred/brent_to_20261109.csv.gz','rt',encoding='utf8') as f:oilbody=json.load(f)['body']
    oil=pd.read_csv(io.StringIO(oilbody),parse_dates=['observation_date'],na_values=['.']).set_index('observation_date').DCOILBRENTEU.dropna().sort_index()
    with gzip.open('data/external/gas/gasradar_daily.csv.gz','rt',encoding='utf8') as f:gasbody=json.load(f)['body']
    gas=pd.read_csv(io.StringIO(gasbody),parse_dates=['gasdag']).set_index('gasdag').eur_per_mwh.dropna().sort_index()
    data=reg.build(raw,weather,oil,gas);cols=list(dict.fromkeys(sum(reg.GROUPS.values(),[])))
    test=data[data.date.between(latest.TEST_START,latest.TEST_END)&np.isfinite(data[cols+['price']]).all(axis=1)]
    saved=joblib.load(reg.OUT/'models.joblib');p=reg.predict(test,saved['models'])
    stored=pd.read_csv(reg.OUT/'predictions.csv.gz',parse_dates=['HourUTC']);checks=0
    for name,values in p.items():
        for zone in reg.ZONES:
            mask=test.zone.eq(zone).to_numpy();q=stored[(stored.model==name)&(stored.zone==zone)]
            if not np.array_equal(test.loc[mask,'HourUTC'].to_numpy(),q.HourUTC.to_numpy()):raise AssertionError('Timestamp mismatch')
            if not np.allclose(values[mask],q.prediction.to_numpy(),rtol=1e-9,atol=1e-9):raise AssertionError('Prediction mismatch')
            checks+=1
    print(f'Reproduced {checks} model/zone groups, {len(test)} zone-hours per model.')


if __name__=='__main__':main()
