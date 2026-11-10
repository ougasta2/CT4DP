"""Conditional error-distribution model fitted only to earlier OOF errors."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_nonlinear as nonlinear

OUT=Path('results/dk1_demand_price')


def choose(samples,reference):
    """Point maximizing estimated eligible ±5% probability in O(k log k)."""
    y=np.asarray(samples);y=y[np.abs(y)>=20]
    if len(y)<10:return float(reference)
    lo=np.sort(y-.05*np.abs(y));hi=np.sort(y+.05*np.abs(y))
    hits=np.searchsorted(lo,lo,side='right')-np.searchsorted(hi,lo,side='left')
    options=lo[hits==hits.max()]
    return float(options[np.argmin(np.abs(options-reference))])


def run():
    p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC').sort_index()
    x=nonlinear.design(p)
    # Standardized only on past training windows. Same-hour neighbor search.
    cols=['base_prediction','DK1_lag1','DK1_lag7','DE-LU_lag1','NL_lag1',
          'temperature_2m','wind_speed_10m','cloud_cover','load_forecast_DK1','load_forecast_DK2',
          'weekday_sin','weekday_cos','year_sin','year_cos']
    x=x[cols]
    if not np.isfinite(x.to_numpy()).all():raise ValueError('incomplete input')
    outputs=[]
    for month in pd.date_range('2023-01-01','2026-01-01',freq='MS',inclusive='left'):
        tr=p.day.lt(month)&p.day.ge(month-pd.Timedelta(days=1095));ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        z=p.loc[ev,['day','hour','DK1','lasso_own_demand_fuel']].copy()
        for k in [50,100,200]:z['knn_'+str(k)]=np.nan
        for hour in range(24):
            train=tr&p.hour.eq(hour);test=ev&p.hour.eq(hour)
            if train.sum()<200 or not test.any():continue
            scaler=StandardScaler().fit(x.loc[train]);xt=scaler.transform(x.loc[train]);xv=scaler.transform(x.loc[test])
            nearest=NearestNeighbors(n_neighbors=200,algorithm='brute').fit(xt)
            idx=nearest.kneighbors(xv,return_distance=False)
            residual=(p.DK1-p.lasso_own_demand_fuel).loc[train].to_numpy()
            ref=p.lasso_own_demand_fuel.loc[test].to_numpy()
            for k in [50,100,200]:
                vals=[choose(r+residual[ix[:k]],r) for r,ix in zip(ref,idx)]
                z.loc[test[test].index,'knn_'+str(k)]=vals
        if z[['knn_50','knn_100','knn_200']].isna().any().any():raise ValueError('incomplete conditional forecast')
        outputs.append(z)
        print('knn',month.date(),flush=True)
    pred=pd.concat(outputs).sort_index();pred.to_csv(OUT/'conditional_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for year,g in pred.groupby(pred.day.dt.year):
        for name in ['lasso_own_demand_fuel','knn_50','knn_100','knn_200']:
            rows.append({'year':year,'model':name,**base.metric(g.DK1,g[name])})
    table=pd.DataFrame(rows);table.to_csv(OUT/'conditional_metrics.csv',index=False)
    dev=table[table.year.isin([2023,2024])];guard=dev[dev.model.eq('lasso_own_demand_fuel')].mae.mean()
    rank=dev.groupby('model').agg(rate=('rate_5pct','mean'),mae=('mae','mean'))
    rank=rank[rank.mae<=guard].sort_values(['rate','mae'],ascending=[False,True]);rank.to_csv(OUT/'conditional_development_ranking.csv')
    (OUT/'conditional_protocol.json').write_text(json.dumps({'selection':'2023/2024 mean hit rate subject to base-lasso mean MAE guardrail',
      'selected':rank.index[0], 'test':'2025 (already partly viewed in previous work)',
      'method':'same-hour neighbors chosen from earlier out-of-fold errors; empirical conditional distribution; maximum estimated +/-5% probability',
      'warning':'Not guaranteed calibrated; historical input publication and revision times unverified.'},indent=2)+'\n',encoding='utf8')
    print('selected',rank.index[0],flush=True);print(table.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
