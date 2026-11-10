"""Reload stored estimators and reproduce all test probabilities."""
import gzip
import io
import json
import joblib
import numpy as np
import pandas as pd
import train_latest_forecast as latest
import train_forecast_models as prior


def main():
    raw=latest.price_pairs();weather={z:latest.weather_all(z) for z in ['DK1','DK2','DE-LU','NL','SE4']}
    with gzip.open('data/external/fred/brent_to_20261109.csv.gz','rt',encoding='utf8') as f:body=json.load(f)['body']
    oil=pd.read_csv(io.StringIO(body),parse_dates=['observation_date'],na_values=['.']).set_index('observation_date').DCOILBRENTEU.dropna().sort_index()
    frame=prior.build_frame(raw,weather,oil);cols=list(dict.fromkeys(sum(prior.FEATURES.values(),[])))
    frame=frame[frame.date.between(latest.TEST_START,latest.TEST_END)&np.isfinite(frame[cols]).all(axis=1)].copy()
    saved=joblib.load(latest.OUT/'models.joblib');stored=pd.read_csv(latest.OUT/'predictions.csv.gz',parse_dates=['HourUTC'])
    checks=0
    for target,obj in saved.items():
        probabilities=latest.predictions(frame,obj['models'],obj['rates'])
        for name,p in probabilities.items():
            for pair in prior.PAIRS:
                mask=frame.pair.eq(pair).to_numpy();expected=stored[(stored.target==target)&(stored.model==name)&(stored.pair==pair)]
                if not np.array_equal(frame.loc[mask,'HourUTC'].to_numpy(),expected.HourUTC.to_numpy()):raise AssertionError('Timestamps differ')
                if not np.allclose(p[mask],expected.p.to_numpy(),rtol=1e-9,atol=1e-9):raise AssertionError('Prediction mismatch')
                checks+=1
    print(f'Reproduced {checks} target/model/pair groups; {len(frame)} pair-hours per model.')


if __name__=='__main__':main()
