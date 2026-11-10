"""Direct hit-probability model for ±5% point forecasts.
Train only on earlier out-of-fold price predictions and labels; random search is
not used. Classification candidates are fixed before test evaluation.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_nonlinear as nonlinear

OUT=Path('results/dk1_demand_price')
DELTAS=np.arange(-.30,.3001,.05)


def expand(x,ref,actual=None):
    n=len(x);k=len(DELTAS)
    values=np.repeat(x.to_numpy(),k,axis=0)
    values=np.column_stack([values,np.tile(DELTAS,n)])
    points=np.repeat(ref,k)+np.tile(DELTAS,n)*np.repeat(np.maximum(np.abs(ref),20),k)
    labels=None if actual is None else np.abs(points-np.repeat(actual,k))<=.05*np.abs(np.repeat(actual,k))
    return values,points,labels


def run():
    past=pd.read_csv(OUT/'nonlinear_predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
    fresh=pd.read_csv(OUT/'evaluation_2026_predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
    p=pd.concat([past,fresh]).sort_index()
    x=nonlinear.design(p);x['hgb_prediction']=p.hgb_price
    cols=['hgb_prediction','base_prediction','DK1_lag1','DK1_lag7','DE-LU_lag1','NL_lag1',
          'temperature_2m','wind_speed_10m','cloud_cover','load_forecast_DK1','load_forecast_DK2',
          'hour_sin','hour_cos','weekday_sin','weekday_cos','year_sin','year_cos']
    x=x[cols];results=[]
    # Deterministic one-third-day sampling controls repeated-candidate training size.
    ordinal=np.array([d.toordinal() for d in p.day])
    for month in pd.date_range('2024-01-01','2026-11-01',freq='MS',inclusive='left'):
        train=p.day.lt(month)&p.day.ge(month-pd.Timedelta(days=730))&p.DK1.abs().ge(20)&(ordinal%3==0)
        ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        xt,_,labels=expand(x.loc[train],p.hgb_price.loc[train].to_numpy(),p.DK1.loc[train].to_numpy())
        model=HistGradientBoostingClassifier(max_iter=100,learning_rate=.04,max_leaf_nodes=15,
                  min_samples_leaf=150,l2_regularization=8.,early_stopping=False,random_state=42)
        model.fit(xt,labels)
        xv,points,_=expand(x.loc[ev],p.hgb_price.loc[ev].to_numpy())
        prob=model.predict_proba(xv)[:,1].reshape(-1,len(DELTAS));points=points.reshape(-1,len(DELTAS))
        # Exact ties prefer zero shift; no true target is used at prediction.
        best=np.argmax(prob-1e-12*np.abs(DELTAS)[None,:],axis=1)
        z=p.loc[ev,['day','hour','DK1','hgb_price']].copy()
        z['hit_probability_point']=points[np.arange(len(z)),best]
        z['estimated_hit_probability']=prob[np.arange(len(z)),best]
        results.append(z);print('hit model',month.date(),flush=True)
    result=pd.concat(results).sort_index();result.to_csv(OUT/'hit_probability_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for year,g in result.groupby(result.day.dt.year):
        for name in ['hgb_price','hit_probability_point']:
            rows.append({'year':year,'model':name,**base.metric(g.DK1,g[name])})
    table=pd.DataFrame(rows);table.to_csv(OUT/'hit_probability_metrics.csv',index=False)
    (OUT/'hit_probability_protocol.json').write_text(json.dumps({'method':'13 fixed candidate shifts -30% to +30%; binary HGB estimates probability of within ±5%; maximum per hour',
     'train':'prior 730 days, only eligible actuals and deterministic one-third of days; underlying price predictions generated OOF',
     'selection':'2024 development; 2025 and 2026 previously viewed in broader work',
     'constraints':'evaluate MAE and negative prices as well as ±5% hit rate; no target-based filtering at prediction time',
     'caveats':['estimated probabilities may be uncalibrated','weather, settlement, price, fuel vintage unverified','15-minute prices averaged hourly']},indent=2)+'\n',encoding='utf8')
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
