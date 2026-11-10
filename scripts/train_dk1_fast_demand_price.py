"""Ablation of fast pre-auction actual-load-derived demand forecasts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_nonlinear as nonlinear

OUT=Path('results/dk1_fast_demand')


def run():
    paths=['results/dk1_demand_price/nonlinear_predictions.csv.gz','results/dk1_demand_price/evaluation_2026_predictions.csv.gz']
    p=pd.concat([pd.read_csv(path,parse_dates=['HourUTC','day']).set_index('HourUTC') for path in paths]).sort_index()
    x=nonlinear.design(p)
    demand=pd.read_csv(OUT/'forecasts.csv.gz',parse_dates=['HourUTC'])
    if demand.duplicated(['HourUTC','zone']).any():raise ValueError('duplicate load forecast')
    fast=demand.pivot(index='HourUTC',columns='zone',values='forecast').reindex(p.index)
    fast.columns=['fast_'+c for c in fast.columns]
    complete=fast[['fast_DK1','fast_DK2']].notna().all(axis=1)&x.notna().all(axis=1)
    p=p.loc[complete].copy();x=x.loc[complete].copy();fast=fast.loc[complete]
    variants={'slow_only':x,'slow_plus_fast':pd.concat([x,fast],axis=1)}
    outputs=[]
    for month in pd.date_range('2023-01-01','2026-08-01',freq='MS',inclusive='left'):
        tr=p.day.lt(month)&p.day.ge(month-pd.Timedelta(days=1095));ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        if tr.sum()<2000 or ev.sum()==0:continue
        z=p.loc[ev,['day','hour','DK1','lasso_own_demand_fuel','previous_day']].copy()
        for name,features in variants.items():
            model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,learning_rate=.04,max_leaf_nodes=15,
                 min_samples_leaf=100,l2_regularization=5.,early_stopping=False,random_state=42)
            model.fit(features.loc[tr],p.DK1.loc[tr]);z[name]=model.predict(features.loc[ev])
        outputs.append(z);print('fast demand ablation',month.date(),flush=True)
    pred=pd.concat(outputs).sort_index();pred.to_csv(OUT/'price_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for year,g in pred.groupby(pred.day.dt.year):
        for name in ['previous_day','lasso_own_demand_fuel','slow_only','slow_plus_fast']:
            rows.append({'year':year,'model':name,**base.metric(g.DK1,g[name])})
    m=pd.DataFrame(rows);m.to_csv(OUT/'price_metrics.csv',index=False)
    print(m.to_string(index=False),flush=True)
    (OUT/'price_protocol.json').write_text(json.dumps({'selection':'slow_only vs slow_plus_fast on 2023/2024; evaluate 2025 and 2026 Jan–Jul on identical available hours',
      'demand':'fast out-of-fold forecasts using 2/7/14-day prior ENTSO-E load',
      'caveats':['missing fast-demand hours are excluded from both comparisons','individual publication times absent despite documented H+1/no revision policy','weather and fuel vintage unverified','2025/2026 previously examined in broader project']},indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
