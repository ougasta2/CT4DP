"""Monthly walk-forward DK1 benchmark: 2023/2024; 2022 calibration.
Weather and commodity release/revision times are not individually verified.
"""
import gzip
import io
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import train_dk1_preauction as base
import train_dk1_weather_ablation as weather_source

OUT=Path('results/dk1_rolling_benchmark')
START=pd.Timestamp('2022-01-01')
END=pd.Timestamp('2025-01-01')
WINDOWS={'8w':56,'12w':84,'3y':1095,'4y':1460}


def price_features(prices, original):
    local=prices.index.tz_convert('Europe/Copenhagen')
    data=prices.copy();data['day']=local.tz_localize(None).normalize();data['hour']=local.hour
    fields=[]
    for z in base.ZONES:
        curve=data.pivot_table(index='day',columns='hour',values=z,aggfunc='mean').reindex(index=original.index,columns=range(24))
        curve=curve.T.interpolate(axis=0,limit_direction='both').T
        for lag in [2,3]:
            f=curve.shift(lag)
            f.columns=[f'{z}_lag{lag}_h{h}' for h in range(24)]
            fields.append(f)
        for lag in [1,7]:
            v=curve.shift(lag)
            f=pd.DataFrame({f'{z}_lag{lag}_mean':v.mean(axis=1),f'{z}_lag{lag}_range':v.max(axis=1)-v.min(axis=1)},index=original.index)
            fields.append(f)
    return pd.concat([original]+fields,axis=1)


def commodity_features(days):
    with gzip.open('data/external/fred/brent_to_20261109.csv.gz','rt',encoding='utf8') as f:
        body=json.load(f)['body']
    oil=pd.read_csv(io.StringIO(body),parse_dates=['observation_date'],na_values=['.']).set_index('observation_date').DCOILBRENTEU.dropna().sort_index()
    with gzip.open('data/external/gas/gasradar_daily.csv.gz','rt',encoding='utf8') as f:
        body=json.load(f)['body']
    gas=pd.read_csv(io.StringIO(body),parse_dates=['gasdag']).set_index('gasdag').eur_per_mwh.dropna().sort_index()
    result=pd.DataFrame(index=days)
    for name,series,lag in [('oil',oil,14),('gas',gas,3)]:
        result[name]=series.reindex(days-pd.Timedelta(days=lag),method='ffill').to_numpy()
        before=series.reindex(days-pd.Timedelta(days=lag+7),method='ffill').to_numpy()
        result[name+'_change7']=result[name]-before
    return result


def fit_model(x,y,kind,asinh=False):
    scale=StandardScaler().fit(x)
    xx=scale.transform(x)
    median=np.median(y.to_numpy(),axis=0)
    spread=np.maximum(np.median(np.abs(y.to_numpy()-median),axis=0)*1.4826,1.)
    yy=(y.to_numpy()-median)/spread
    if asinh: yy=np.arcsinh(yy)
    if kind=='ridge': model=Ridge(alpha=100.)
    elif kind=='lasso':model=Lasso(alpha=.03,max_iter=1200,tol=.002)
    elif kind=='mlp': model=MLPRegressor(hidden_layer_sizes=(80,40),alpha=1.,max_iter=120,random_state=42,shuffle=True,early_stopping=False)
    else:raise ValueError(kind)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning)
        model.fit(xx,yy)
    return {'scaler':scale,'model':model,'median':median,'spread':spread,'asinh':asinh}


def predict_model(model,x):
    p=model['model'].predict(model['scaler'].transform(x))
    if model['asinh']:p=np.sinh(np.clip(p,-8,8))
    return p*model['spread']+model['median']


def best_interval_prediction(samples,reference):
    """Maximize hit count for eligible sample outcomes; closest to reference tie."""
    s=np.asarray(samples);s=s[np.abs(s)>=20]
    if len(s)<10:return float(reference)
    lo=s-.05*np.abs(s);hi=s+.05*np.abs(s)
    # Each optimum is attained at a left endpoint; evaluating those points also
    # preserves the inclusive <= boundary in the actual metric.
    hits=((lo[:,None]<=lo[None,:]) & (hi[:,None]>=lo[None,:])).sum(axis=0)
    candidates=lo[hits==hits.max()]
    return float(candidates[np.argmin(np.abs(candidates-reference))])


def apply_distribution(p):
    """Online calibration from previous predictions; no residuals from today."""
    p=p.sort_index().copy()
    p['interval_point']=np.nan;p['q05']=np.nan;p['q95']=np.nan;p['crps']=np.nan
    for day in sorted(p.day.unique()):
        day=pd.Timestamp(day)
        # At D-1 10:00, day-ahead clearing results for D-1 normally known
        # from its auction D-2; exceptional publication delays remain unaudited.
        hist=p[(p.day<day)&(p.day>=day-pd.Timedelta(days=365))]
        for i,row in p[p.day.eq(day)].iterrows():
            residual=(hist.loc[hist.hour.eq(row.hour),'DK1']-hist.loc[hist.hour.eq(row.hour),'lear_ensemble']).dropna().to_numpy()
            if len(residual)<30:continue
            samples=row.lear_ensemble+residual
            p.at[i,'interval_point']=best_interval_prediction(samples,row.lear_ensemble)
            p.at[i,'q05'],p.at[i,'q95']=np.quantile(samples,[.05,.95])
            # empirical CRPS identity using sorted samples, O(n log n)
            s=np.sort(samples);n=len(s)
            p.at[i,'crps']=np.mean(np.abs(s-row.DK1))-np.sum((2*np.arange(1,n+1)-n-1)*s)/(n*n)
    return p


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    prices=base.load_prices()
    orig,y,data=base.make_daily(prices)
    extra=price_features(prices,orig)
    weather,coverage=weather_source.weather_curves(prices)
    fuel=commodity_features(orig.index)
    inputs={'original_weather':pd.concat([orig,weather],axis=1),
            'extended_weather':pd.concat([extra,weather],axis=1),
            'extended_weather_fuel':pd.concat([extra,weather,fuel],axis=1)}
    full=inputs['extended_weather_fuel']
    counts=data.groupby('day').size().reindex(orig.index)
    expected=pd.Series([(pd.Timestamp(d+pd.Timedelta(days=1),tz='Europe/Copenhagen').tz_convert('UTC')-pd.Timestamp(d,tz='Europe/Copenhagen').tz_convert('UTC')).total_seconds()/3600 for d in orig.index],index=orig.index)
    valid=full.notna().all(axis=1)&y.notna().all(axis=1)&counts.eq(expected)
    days=orig.index[valid]
    allp=[];snapshots={}
    for month in pd.date_range(START,END,freq='MS',inclusive='left'):
        end=month+pd.offsets.MonthBegin(1)
        ev=days[(days>=month)&(days<end)]
        train=days[days<month]
        if len(ev)!=(end-month).days:raise ValueError(f'Missing evaluation days in {month}')
        target=data[data.day.isin(ev)][['day','hour','DK1']].copy()
        if target.empty:raise ValueError('Empty month')
        lookup=pd.DatetimeIndex(target.day); h=target.hour.to_numpy()
        target['previous_day']=orig.loc[lookup].to_numpy()[np.arange(len(target)),h]
        jobs=[('ridge_original_weather','original_weather','ridge',None,False),
              ('ridge_extended_weather','extended_weather','ridge',None,False),
              ('ridge_extended_weather_fuel','extended_weather_fuel','ridge',None,False)]
        jobs.extend((f'lear_{name}','extended_weather_fuel','lasso',days_window,True) for name,days_window in WINDOWS.items())
        jobs.append(('lear_raw_3y','extended_weather_fuel','lasso',1095,False))
        if month.month==1:
            mlp=fit_model(inputs['extended_weather_fuel'].loc[train],y.loc[train],'mlp')
        for name,features,kind,window,asinh in jobs:
            tr=train if window is None else train[train>=month-pd.Timedelta(days=window)]
            model=fit_model(inputs[features].loc[tr],y.loc[tr],kind,asinh)
            curve=predict_model(model,inputs[features].loc[ev])
            target[name]=pd.DataFrame(curve,index=ev).loc[lookup].to_numpy()[np.arange(len(target)),h]
            if month==pd.Timestamp('2024-12-01'):
                snapshots[name]={'estimator':model,'feature_columns':list(inputs[features].columns),'trained_through':str(train[-1].date()),'window_days':window}
        curve=predict_model(mlp,inputs['extended_weather_fuel'].loc[ev])
        target['mlp_annual']=pd.DataFrame(curve,index=ev).loc[lookup].to_numpy()[np.arange(len(target)),h]
        target['lear_ensemble']=target[[f'lear_{name}' for name in WINDOWS]].mean(axis=1)
        allp.append(target)
        print('completed',month.date(), 'train',len(train),'eval',len(ev),flush=True)
    pred=apply_distribution(pd.concat(allp).sort_index())
    pred.to_csv(OUT/'predictions.csv.gz',compression={'method':'gzip','mtime':0})
    model_names=['previous_day','ridge_original_weather','ridge_extended_weather','ridge_extended_weather_fuel',
                 *[f'lear_{name}' for name in WINDOWS],'lear_ensemble','lear_raw_3y','mlp_annual','interval_point']
    rows=[];segments=[]
    for year,g in pred[pred.day.dt.year.isin([2023,2024])].groupby(pred.day.dt.year):
        for name in model_names:
            z=g[g[name].notna()]
            rows.append({'year':year,'model':name,**base.metric(z.DK1,z[name])})
            for label,mask in [('negative',z.DK1<0),('high_200_plus',z.DK1>=200),('afternoon',z.hour.between(12,17))]:
                zz=z[mask]
                if len(zz):segments.append({'year':year,'model':name,'segment':label,**base.metric(zz.DK1,zz[name])})
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(segments).to_csv(OUT/'segments.csv',index=False)
    q=pred[(pred.day.dt.year.isin([2023,2024]))&pred.q05.notna()]
    prob=q.groupby(q.day.dt.year).apply(lambda g:pd.Series({'hours':len(g),'coverage_90':((g.DK1>=g.q05)&(g.DK1<=g.q95)).mean(),'mean_width_90':(g.q95-g.q05).mean(),'crps':g.crps.mean()}))
    prob.to_csv(OUT/'distribution_metrics.csv')
    joblib.dump({'december_2024':snapshots,'annual_mlp':mlp,'mlp_columns':list(inputs['extended_weather_fuel'].columns)},OUT/'models.joblib',compress=3)
    protocol={'evaluation':'2023 and 2024 calendar years; all model exploration','calibration_predictions':'2022, chronological monthly fits',
      'cutoff':'10:00 Copenhagen on D-1', 'recalibration':'monthly; 4 rolling-window Lasso variants; annual MLP',
      'LEAR':'independent per-hour Lasso, fixed alpha 0.03 after standardization, asinh unless raw variant; inspired by LEAR, not exact epftoolbox',
      'probabilistic':'last 365 days of past same-hour ensemble errors, minimum 30, empirical samples; not guaranteed calibrated',
      'fuels':'Gasradar EnergyZero gas-day proxy lag 3d, FRED Brent lag 14d; not audited point-in-time',
      'weather':'Open-Meteo JMA previous_day2 48h archive; release and revision timestamps not verified',
      'limitations':['Forecasts/inputs not validated as published by cutoff; results exploratory.', '10:00 prior-day power-price publication assumed; delay/revisions unverified.', '2023 and 2024 are used for benchmarking and selection.', 'Hourly averages; repeated DST hour averaged and spring hour interpolated.', 'monthly recalibration differs from daily literature and equal-weight ensemble is fixed.', 'No validated load/renewable forecast vintages.'],
      'sampling':'all models use identical complete days for weather and fuel even price baseline'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
