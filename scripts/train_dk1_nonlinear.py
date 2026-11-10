"""Nonlinear price forecasts using honest historical demand forecasts and residuals.

Gradient-boosted decision trees fit either price or *out-of-fold* residuals;
they never train on residuals computed from in-sample base predictions.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_latest_forecast as latest
from train_price_regression import local_lag

OUT=Path('results/dk1_demand_price')


def design(pred):
    times=pd.DatetimeIndex(pred.index)
    prices=base.load_prices()
    weather=latest.weather_all('DK1').reindex(times)
    forecast=pd.read_csv('results/dk1_demand_forecast/demand_forecasts.csv.gz',parse_dates=['HourUTC'])
    f=forecast.pivot(index='HourUTC',columns='zone',values='forecast').reindex(times)
    x=pd.DataFrame(index=times)
    for z in base.ZONES:
        full=prices[z]
        fz=pd.DataFrame({'HourUTC':full.index,'price':full.to_numpy()})
        x[z+'_lag1']=pd.Series(local_lag(fz,1),index=full.index).reindex(times).to_numpy()
        x[z+'_lag7']=pd.Series(local_lag(fz,7),index=full.index).reindex(times).to_numpy()
    for c in ['temperature_2m','wind_speed_10m','cloud_cover']:x[c]=weather[c].to_numpy()
    for zone in ['DK1','DK2']:x['load_forecast_'+zone]=f[zone].to_numpy()
    x['base_prediction']=pred.lasso_own_demand_fuel.to_numpy()
    day=pd.DatetimeIndex(pred.day)
    x['hour_sin']=np.sin(2*np.pi*pred.hour.to_numpy()/24);x['hour_cos']=np.cos(2*np.pi*pred.hour.to_numpy()/24)
    x['weekday_sin']=np.sin(2*np.pi*day.dayofweek/7);x['weekday_cos']=np.cos(2*np.pi*day.dayofweek/7)
    x['year_sin']=np.sin(2*np.pi*day.dayofyear/365.25);x['year_cos']=np.cos(2*np.pi*day.dayofyear/365.25)
    return x


def run():
    p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC').sort_index()
    x=design(p)
    if not np.isfinite(x.to_numpy()).all():raise ValueError('Missing explanatory inputs')
    pred=[]
    for month in pd.date_range('2023-01-01','2026-01-01',freq='MS',inclusive='left'):
        train=p.day.lt(month)&p.day.ge(month-pd.Timedelta(days=1095))
        ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        if train.sum()<2000 or ev.sum()==0:continue
        z=p.loc[ev,['day','hour','DK1','previous_day','lasso_own_demand_fuel']].copy()
        for objective,loss in [('price','absolute_error'),('residual','absolute_error'),('residual_mse','squared_error')]:
            model=HistGradientBoostingRegressor(loss=loss,max_iter=100,learning_rate=.04,max_leaf_nodes=15,
                min_samples_leaf=100,l2_regularization=5.,early_stopping=False,random_state=42)
            y=p.DK1.loc[train] if objective=='price' else (p.DK1-p.lasso_own_demand_fuel).loc[train]
            model.fit(x.loc[train],y)
            v=model.predict(x.loc[ev])
            z['hgb_'+objective]=v if objective=='price' else z.lasso_own_demand_fuel+v
        z['hgb_residual_blend']=.5*z.hgb_residual+.5*z.lasso_own_demand_fuel
        pred.append(z)
        print('nonlinear',month.date(),flush=True)
    pred=pd.concat(pred).sort_index()
    pred.to_csv(OUT/'nonlinear_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for year,g in pred.groupby(pred.day.dt.year):
        for c in ['previous_day','lasso_own_demand_fuel','hgb_price','hgb_residual','hgb_residual_mse','hgb_residual_blend']:
            rows.append({'year':year,'model':c,**base.metric(g.DK1,g[c])})
    table=pd.DataFrame(rows);table.to_csv(OUT/'nonlinear_metrics.csv',index=False)
    dev=table[table.year.isin([2023,2024])]
    guard=dev[dev.model.eq('lasso_own_demand_fuel')].mae.mean()
    ranked=dev.groupby('model').agg(rate=('rate_5pct','mean'),mae=('mae','mean'))
    ranked=ranked[ranked.mae<=guard].sort_values(['rate','mae'],ascending=[False,True]);ranked.to_csv(OUT/'nonlinear_development_ranking.csv')
    print('selected',ranked.index[0],flush=True);print(table.to_string(index=False),flush=True)
    (OUT/'nonlinear_protocol.json').write_text(json.dumps({'development':'2023 and 2024','evaluation':'2025 previously studied, not pristine',
      'selection':'best dev mean +/-5% under base lasso mean MAE guardrail','selected':ranked.index[0],
      'residuals':'only past monthly out-of-fold predictions from demand-price models',
      'caveats':['Issue time assumptions of weather, settlement, prices and fuel not individually verified','hourly average prices, not 15-min predictions']},indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
