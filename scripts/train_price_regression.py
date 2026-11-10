"""Predict hourly area prices with point-in-time proxy inputs and temporal holdout."""
import gzip
import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from energy_pilot.mining import apriori,mask_for

import train_latest_forecast as latest
import train_forecast_models as prior
from evaluate_price_signs import write_sign_metrics

OUT=Path('results/price_forecast');ZONES=['DK1','DK2','DE-LU','NL','SE4']
CAL=['hour_sin','hour_cos','month_sin','month_cos','weekday_sin','weekday_cos']+[f'zone_{z}' for z in ZONES]+['de_nuclear_closed']
PRICES=['price_previous_local_day','price_previous_week']
WEATHER=['temperature_2m','wind_speed_10m','cloud_cover']
GROUPS={'price_history':CAL+PRICES,'weather':CAL+PRICES+WEATHER,
        'weather_oil':CAL+PRICES+WEATHER+['brent_14d'],
        'weather_gas':CAL+PRICES+WEATHER+['gas_3d'],
        'weather_gas_oil':CAL+PRICES+WEATHER+['gas_3d','brent_14d']}


def local_lag(frame,days):
    """Previous local-day/hour market price; average repeated hour at fall DST."""
    local=frame.HourUTC.dt.tz_convert('Europe/Copenhagen')
    dates=pd.to_datetime(local.dt.date)
    cells=pd.DataFrame({'date':dates,'hour':local.dt.hour,'price':frame.price.to_numpy()})
    mapping=cells.groupby(['date','hour']).price.mean()
    key=pd.MultiIndex.from_arrays([dates-pd.Timedelta(days=days),local.dt.hour])
    result=mapping.reindex(key).to_numpy()
    # A spring DST missing clock hour falls back to an earlier UTC price.
    fallback=frame.price.shift(days*24+24).to_numpy()
    return np.where(np.isfinite(result),result,fallback)


def build(raw,weather,oil,gas):
    pieces=[]
    for zone in ZONES:
        candidates=[]
        for pair in prior.PAIRS:
            a,b=pair.split('--');g=raw[raw.pair.eq(pair)]
            if a==zone:candidates.append(g[['HourUTC','price_dk']].rename(columns={'price_dk':'price'}))
            if b==zone:candidates.append(g[['HourUTC','price_neighbor']].rename(columns={'price_neighbor':'price'}))
        x=pd.concat(candidates,ignore_index=True)
        if (x.groupby('HourUTC').price.nunique()>1).any():raise ValueError('Inconsistent duplicated zone prices')
        x=x.drop_duplicates('HourUTC').sort_values('HourUTC').reset_index(drop=True)
        if not x.HourUTC.diff().dropna().eq(pd.Timedelta(hours=1)).all():raise ValueError('Missing zone price hour')
        x['zone']=zone;x['price_previous_local_day']=local_lag(x,1);x['price_previous_week']=local_lag(x,7)
        local=x.HourUTC.dt.tz_convert('Europe/Copenhagen');x['date']=local.dt.date
        for name,val,period in [('hour',local.dt.hour,24),('month',local.dt.month-1,12),('weekday',local.dt.weekday,7)]:
            x[name+'_sin']=np.sin(2*np.pi*val/period);x[name+'_cos']=np.cos(2*np.pi*val/period)
        for z in ZONES:x['zone_'+z]=int(z==zone)
        x['de_nuclear_closed']=(local.dt.date>=pd.Timestamp('2023-04-16').date()).astype(int)
        w=weather[zone].reindex(pd.DatetimeIndex(x.HourUTC))
        for name in WEATHER:x[name]=w[name].to_numpy()
        day=pd.DatetimeIndex(pd.to_datetime(x.date))
        x['brent_14d']=oil.reindex(day-pd.Timedelta(days=14),method='ffill').to_numpy()
        x['gas_3d']=gas.reindex(day-pd.Timedelta(days=3),method='ffill').to_numpy()
        issue=x.HourUTC-pd.Timedelta(hours=48)
        cutoff=(pd.to_datetime(x.date)-pd.Timedelta(days=1)+pd.Timedelta(hours=10)).dt.tz_localize('Europe/Copenhagen').dt.tz_convert('UTC')
        if not (issue.to_numpy()<=cutoff.to_numpy()).all():raise ValueError('Forecast cutoff violation')
        pieces.append(x)
    return pd.concat(pieces,ignore_index=True)


def score(y,p):
    error=np.asarray(p)-np.asarray(y); abs_err=np.abs(error);qualified=np.abs(y)>=20
    within=abs_err[qualified]<=.03*np.abs(y[qualified])
    return dict(n=len(y),mae=float(abs_err.mean()),rmse=float(np.sqrt(np.mean(error**2))),
                median_ae=float(np.median(abs_err)),mean_error=float(error.mean()),
                percent3_eligible=int(qualified.sum()),percent3_hits=int(within.sum()),
                percent3_rate=float(within.mean()) if len(within) else None)


def model(family):
    if family=='ridge':return make_pipeline(StandardScaler(),Ridge(alpha=100.))
    return HistGradientBoostingRegressor(loss='squared_error',max_iter=150,learning_rate=.05,max_leaf_nodes=15,
         min_samples_leaf=100,l2_regularization=1.,early_stopping=False,random_state=42)


def fit_all(train):
    models={};sizes=train.groupby('zone').size().to_dict()
    w=train.zone.map({z:len(train)/(len(ZONES)*n) for z,n in sizes.items()}).to_numpy()
    for family in ['ridge','boosting']:
        for group,features in GROUPS.items():
            print('Fitting',family,group,flush=True);est=model(family)
            kw={'ridge__sample_weight':w} if family=='ridge' else {'sample_weight':w}
            est.fit(train[features],train.price,**kw);models[family+'_'+group]=est
    return models


def predict(x,models):
    predictions={'previous_day':x.price_previous_local_day.to_numpy()}
    for name,clf in models.items():predictions[name]=clf.predict(x[GROUPS[name.split('_',1)[1]]])
    return predictions


def interval(frame,delta):return latest.day_interval(frame,delta,repeats=1000,block=7)


def error_baskets(frame):
    local=frame.HourUTC.dt.tz_convert('Europe/Copenhagen')
    season=np.select([local.dt.month.isin([12,1,2]),local.dt.month.isin([3,4,5]),
                      local.dt.month.isin([6,7,8])],['winter','spring','summer'],default='autumn')
    times=pd.cut(local.dt.hour,[-1,5,11,17,23],labels=['night','morning','afternoon','evening']).astype(str)
    return [frozenset([f'cluster=C{c}',f'season={s}',f'time={t}']) for c,s,t in zip(frame.cluster,season,times)]


def fit_error_rules(predev,dev,test,dev_prediction,test_prediction):
    # Unsupervised fit excludes current target prices; rules use only dev residuals.
    fields=PRICES+WEATHER;scale=StandardScaler().fit(predev[fields])
    cluster=KMeans(n_clusters=4,n_init=10,random_state=42).fit(scale.transform(predev[fields]))
    dev=dev.copy();test=test.copy();dev['cluster']=cluster.predict(scale.transform(dev[fields]));test['cluster']=cluster.predict(scale.transform(test[fields]))
    dev['absolute_error']=np.abs(dev.price.to_numpy()-dev_prediction)
    test['absolute_error']=np.abs(test.price.to_numpy()-test_prediction)
    rows=[]
    for zone,g in dev.groupby('zone'):
        threshold=float(g.absolute_error.quantile(.90));events=(g.absolute_error>threshold).to_numpy();basket=error_baskets(g);options=[]
        for ant,n in apriori(basket,min_support=.05,max_size=3).items():
            if not any(v.startswith('cluster=') for v in ant):continue
            mask=mask_for(basket,ant);count=int(events[mask].sum())
            if n>=100 and count>=20 and events.mean()>0:options.append((count/n/events.mean(),n,tuple(sorted(ant)),count))
        if not options:continue
        lift,n,ant,count=max(options);z=test[test.zone==zone];e=(z.absolute_error>threshold).to_numpy()
        match=mask_for(error_baskets(z),frozenset(ant));test_n=int(match.sum());test_events=int(e[match].sum());base=float(e.mean())
        rows.append(dict(zone=zone,antecedent=' & '.join(ant),threshold_eur_mwh=threshold,
                         dev_n=n,dev_events=count,dev_lift=lift,test_n=test_n,test_events=test_events,
                         test_rate=test_events/test_n if test_n else np.nan,test_base_rate=base,
                         test_lift=test_events/test_n/base if test_n and base else np.nan))
    centers=pd.DataFrame(scale.inverse_transform(cluster.cluster_centers_),columns=fields);centers.index.name='cluster'
    return pd.DataFrame(rows),centers,scale,cluster


def main():
    OUT.mkdir(parents=True,exist_ok=True);raw=latest.price_pairs();weather={z:latest.weather_all(z) for z in ZONES}
    with gzip.open('data/external/fred/brent_to_20261109.csv.gz','rt',encoding='utf8') as f:oilbody=json.load(f)['body']
    oil=pd.read_csv(io.StringIO(oilbody),parse_dates=['observation_date'],na_values=['.']).set_index('observation_date').DCOILBRENTEU.dropna().sort_index()
    with gzip.open('data/external/gas/gasradar_daily.csv.gz','rt',encoding='utf8') as f:gasbody=json.load(f)['body']
    gas=pd.read_csv(io.StringIO(gasbody),parse_dates=['gasdag']).set_index('gasdag').eur_per_mwh.dropna().sort_index()
    if oil.index.has_duplicates or gas.index.has_duplicates:raise ValueError('Duplicate commodity observation date')
    data=build(raw,weather,oil,gas);features=list(dict.fromkeys(sum(GROUPS.values(),[])))
    data['complete']=np.isfinite(data[features+['price']]).all(axis=1)
    data.groupby(['zone',data.HourUTC.dt.tz_convert('Europe/Copenhagen').dt.year]).agg(hours=('complete','size'),complete=('complete','sum')).to_csv(OUT/'input_coverage.csv')
    d=data[data.complete].copy();predev=d[d.date<latest.DEV_START];dev=d[d.date.between(latest.DEV_START,latest.DEV_END)]
    train=d[d.date<latest.TEST_START];test=d[d.date.between(latest.TEST_START,latest.TEST_END)]
    expected=pd.date_range(pd.Timestamp(latest.TEST_START,tz='Europe/Copenhagen').tz_convert('UTC'),
          (pd.Timestamp(latest.TEST_END,tz='Europe/Copenhagen')+pd.DateOffset(days=1)).tz_convert('UTC'),freq='h',inclusive='left')
    for zone,g in test.groupby('zone'):
        if not pd.DatetimeIndex(g.HourUTC).equals(expected):raise ValueError(f'Incomplete test hours for {zone}')
    if test.zone.nunique()!=len(ZONES):raise ValueError('Missing test zone')
    print('Fitting development models',flush=True);devmodels=fit_all(predev);p=predict(dev,devmodels);devrows=[]
    for (year,zone),g in dev.groupby([dev.HourUTC.dt.tz_convert('Europe/Copenhagen').dt.year,'zone']):
        i=dev.index.get_indexer(g.index)
        for name,values in p.items():devrows.append(dict(year=int(year),zone=zone,model=name,**score(g.price.to_numpy(),values[i])))
    devtable=pd.DataFrame(devrows);devtable.to_csv(OUT/'development_metrics.csv',index=False)
    ranking=devtable.groupby('model').mae.mean().sort_values();selected=str(ranking.index[0])
    print('Fitting final models, selected:',selected,flush=True);models=fit_all(train);probs=predict(test,models)
    rules,centers,cluster_scaler,cluster_model=fit_error_rules(predev,dev,test,p[selected],probs[selected])
    rules.to_csv(OUT/'error_rules.csv',index=False);centers.to_csv(OUT/'cluster_centers.csv')
    rows=[];preds=[];intervals=[];segments=[];regimes=[]
    gas_p75=float(train.gas_3d.quantile(.75))
    for zone,g in test.groupby('zone'):
        ix=test.index.get_indexer(g.index);y=g.price.to_numpy()
        for name,values in probs.items():
            yp=values[ix];rows.append(dict(zone=zone,model=name,**score(y,yp)))
            preds.append(pd.DataFrame({'HourUTC':g.HourUTC,'zone':zone,'model':name,'actual':y,'prediction':yp}))
            masks={'negative':y<0,'zero_to_20':(y>=0)&(y<20),'20_to_100':(y>=20)&(y<100),
                   '100_to_200':(y>=100)&(y<200),'200_plus':y>=200}
            for segment,mask in masks.items():
                if mask.any():segments.append(dict(zone=zone,model=name,segment=segment,**score(y[mask],yp[mask])))
            for label,mask in [('gas_at_or_above_train_p75',(g.gas_3d>=gas_p75).to_numpy()),
                               ('gas_below_train_p75',(g.gas_3d<gas_p75).to_numpy())]:
                if mask.any():regimes.append(dict(zone=zone,model=name,regime=label,gas_threshold_eur_mwh=gas_p75,
                                                   mean_actual_price=float(y[mask].mean()),**score(y[mask],yp[mask])))
        for name,baseline in [(selected,'previous_day'),('boosting_weather','boosting_price_history'),
                              ('boosting_weather_gas','boosting_weather'),('boosting_weather_oil','boosting_weather'),
                              ('boosting_weather_gas_oil','boosting_weather')]:
            delta=np.abs(probs[baseline][ix]-y)-np.abs(probs[name][ix]-y);lo,hi=interval(g,delta)
            intervals.append(dict(zone=zone,model=name,baseline=baseline,mae_improvement=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),block_days=7))
    pd.DataFrame(rows).to_csv(OUT/'test_metrics.csv',index=False)
    pd.DataFrame(segments).to_csv(OUT/'price_band_metrics.csv',index=False)
    pd.DataFrame(regimes).to_csv(OUT/'gas_regime_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'mae_intervals.csv',index=False)
    predictions=pd.concat(preds,ignore_index=True);predictions.to_csv(OUT/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    write_sign_metrics(predictions,OUT)
    predictions['local_year']=pd.to_datetime(predictions.HourUTC,utc=True).dt.tz_convert('Europe/Copenhagen').dt.year
    byyear=[dict(zone=z,model=m,local_year=int(yr),**score(g.actual.to_numpy(),g.prediction.to_numpy()))
            for (z,m,yr),g in predictions.groupby(['zone','model','local_year'])]
    pd.DataFrame(byyear).to_csv(OUT/'test_by_calendar_year.csv',index=False)
    joblib.dump({'models':models,'selected':selected,'error_cluster_scaler':cluster_scaler,'error_clusters':cluster_model},OUT/'models.joblib',compress=3)
    protocol=dict(train_start=str(min(train.date)),train_end='2025-11-09',train_hours_by_zone=train.groupby('zone').size().to_dict(),
       development='2024-01-01 to 2025-11-09, fit through 2023',test='2025-11-10 to 2026-11-09',
       target='area hourly average day-ahead price EUR/MWh',selection='minimum equally weighted mean zone/year MAE on development',
       selected=selected,development_mean_mae=ranking.to_dict(),features=GROUPS,training_gas_p75_eur_mwh=gas_p75,
       assumptions=['weather archived 48h-lead issue time not individually verified','Brent 14-day observation lag, historical publication and revision unverified',
                    'EnergyZero-derived Gasradar gas-day EUR/MWh proxy, 3-day date lag, historical publication and revisions unverified',
                    'nuclear post-shutdown indicator is not causal and cannot account for unobserved shocks',
                    '2025 first 52 test days previously inspected'],seed=42)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
