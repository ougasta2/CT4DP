"""Conditional ±5% point selection around the nonlinear OOF predictor."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_nonlinear as nonlinear
from train_dk1_conditional import choose

OUT=Path('results/dk1_demand_price')


def run():
    past=pd.read_csv(OUT/'nonlinear_predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
    fresh=pd.read_csv(OUT/'evaluation_2026_predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
    p=pd.concat([past,fresh]).sort_index()
    x=nonlinear.design(p)
    x['hgb_prediction']=p.hgb_price
    cols=['hgb_prediction','base_prediction','DK1_lag1','DK1_lag7','DE-LU_lag1','NL_lag1',
          'temperature_2m','wind_speed_10m','cloud_cover','load_forecast_DK1','load_forecast_DK2',
          'weekday_sin','weekday_cos','year_sin','year_cos']
    x=x[cols];outputs=[]
    for month in pd.date_range('2024-01-01','2026-11-01',freq='MS',inclusive='left'):
        tr=p.day.lt(month)&p.day.ge(month-pd.Timedelta(days=1095));ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        z=p.loc[ev,['day','hour','DK1','hgb_price']].copy()
        for k in [50,100,200]:z['knn_'+str(k)]=np.nan
        for hour in range(24):
            train=tr&p.hour.eq(hour);test=ev&p.hour.eq(hour)
            if train.sum()<200 or not test.any():continue
            scaler=StandardScaler().fit(x.loc[train]);xt=scaler.transform(x.loc[train]);xv=scaler.transform(x.loc[test])
            neighbors=NearestNeighbors(n_neighbors=200,algorithm='brute').fit(xt).kneighbors(xv,return_distance=False)
            residual=(p.DK1-p.hgb_price).loc[train].to_numpy();ref=p.hgb_price.loc[test].to_numpy()
            for k in [50,100,200]:z.loc[test[test].index,'knn_'+str(k)]=[choose(r+residual[ix[:k]],r) for r,ix in zip(ref,neighbors)]
        if z[['knn_50','knn_100','knn_200']].isna().any().any():raise ValueError('incomplete forecast')
        outputs.append(z);print('hgb conditional',month.date(),flush=True)
    r=pd.concat(outputs).sort_index();r.to_csv(OUT/'hgb_conditional_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for year,g in r.groupby(r.day.dt.year):
        for name in ['hgb_price','knn_50','knn_100','knn_200']:
            rows.append({'year':year,'model':name,**base.metric(g.DK1,g[name])})
    table=pd.DataFrame(rows);table.to_csv(OUT/'hgb_conditional_metrics.csv',index=False)
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
