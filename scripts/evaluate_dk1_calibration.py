"""Chronological error-bias corrections using only earlier predictions/outcomes."""
from pathlib import Path
import numpy as np
import pandas as pd
import train_dk1_preauction as base

OUT=Path('results/dk1_rolling_benchmark')


def add_corrections(df):
    p=df.sort_index().copy()
    p['blend_50']=(p.ridge_original_weather+p.lear_raw_3y)/2
    for model in ['ridge_original_weather','lear_raw_3y','blend_50']:
        # Aggregate the repeated autumn hour before shifting, so the second
        # occurrence can never use the first occurrence's same-day outcome.
        residual=pd.DataFrame({'day':p.day,'hour':p.hour,'residual':p.DK1-p[model]}).groupby(['hour','day']).residual.mean()
        for window in [30,90]:
            for stat in ['mean','median']:
                correction=pd.Series(np.nan,index=p.index)
                for hour in range(24):
                    mask=p.hour.eq(hour)
                    h=residual.loc[hour].shift(1)
                    offset=getattr(h.rolling(window,min_periods=15),stat)().fillna(0)
                    correction.loc[mask]=p.loc[mask,'day'].map(offset).to_numpy()
                p[f'{model}_{stat}{window}']=p[model]+correction
    return p


def main():
    p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
    p=add_corrections(p)
    columns=[c for c in p.columns if c.startswith(('ridge_original_weather','lear_raw_3y','blend_50'))]
    rows=[]
    for year,g in p[p.day.dt.year.isin([2023,2024])].groupby(p.day.dt.year):
        for col in columns:rows.append({'year':year,'model':col,**base.metric(g.DK1,g[col])})
    result=pd.DataFrame(rows);result.to_csv(OUT/'calibration_metrics.csv',index=False)
    p.to_csv(OUT/'calibrated_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    print(result.to_string(index=False))

if __name__=='__main__':main()
