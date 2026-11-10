"""Matched ablation for pre-auction proxy wind, solar and residual load."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_nonlinear as nonlinear

OUT=Path('results/dk1_demand_price')


def supply_features(index,x,frame=None):
    if frame is None:frame=pd.read_csv('results/dk1_supply_forecast/forecasts.csv.gz',parse_dates=['HourUTC'])
    frame=frame.copy()
    if frame.duplicated(['HourUTC','zone']).any():raise ValueError('duplicate supply forecast')
    # Keep predicted fields only. Actual generation columns are never features.
    extra=pd.DataFrame(index=index)
    for zone in ['DK1','DK2']:
        z=frame[frame.zone.eq(zone)].set_index('HourUTC').reindex(index)
        for tech in ['wind_mwh','solar_harmonized_total_mwh']:
            extra[f'{zone}_{tech}_forecast']=z['forecast_'+tech].to_numpy()
        extra[f'{zone}_residual_load']=x['load_forecast_'+zone]-extra[f'{zone}_wind_mwh_forecast']-extra[f'{zone}_solar_harmonized_total_mwh_forecast']
    return extra


def run():
    p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC').sort_index()
    x=nonlinear.design(p);extra=supply_features(p.index,x)
    if not np.isfinite(extra.to_numpy()).all():raise ValueError('Missing supply forecasts')
    variants={'without_supply':x,'with_supply':pd.concat([x,extra],axis=1)}
    outputs=[]
    for month in pd.date_range('2023-01-01','2026-01-01',freq='MS',inclusive='left'):
        train=p.day.lt(month)&p.day.ge(month-pd.Timedelta(days=1095));ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        z=p.loc[ev,['day','hour','DK1']].copy()
        for name,xx in variants.items():
            model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,learning_rate=.04,max_leaf_nodes=15,
               min_samples_leaf=100,l2_regularization=5.,early_stopping=False,random_state=42)
            model.fit(xx.loc[train],p.DK1.loc[train]);z[name]=model.predict(xx.loc[ev])
        outputs.append(z);print('supply ablation',month.date(),flush=True)
    result=pd.concat(outputs).sort_index();result.to_csv(OUT/'supply_ablation_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for year,g in result.groupby(result.day.dt.year):
        for name in variants:rows.append({'year':year,'model':name,**base.metric(g.DK1,g[name])})
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'supply_ablation_metrics.csv',index=False)
    dev=metrics[metrics.year.isin([2023,2024])].groupby('model').agg(rate=('rate_5pct','mean'),mae=('mae','mean'))
    guard=dev.loc['without_supply','mae'];dev=dev[dev.mae<=guard].sort_values(['rate','mae'],ascending=[False,True])
    print('selection',dev.index[0],flush=True);print(metrics.to_string(index=False),flush=True)
    (OUT/'supply_protocol.json').write_text(json.dumps({'selected':dev.index[0],
       'selection':'2023–2024 mean hit rate, MAE no worse than without-supply baseline',
       'evaluation':'2025, partly seen in prior work','actual_generation_used':'only to train past out-of-fold supply forecasting models; never target-day price-model features',
       'limitations':['publication and revision timestamps of underlying settlement and forecast weather unverified','2025-10 onwards hourly averages from 15-minute prices']},indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
