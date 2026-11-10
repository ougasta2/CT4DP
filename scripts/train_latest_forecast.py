"""Freeze models on past data and score the latest 365 complete local days."""
import gzip
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import prepare_multiarea as source
import train_forecast_models as prior

OUT=Path('results/latest_forecast')
TEST_START=pd.Timestamp('2025-11-10').date()
TEST_END=pd.Timestamp('2026-11-09').date()
DEV_START=pd.Timestamp('2024-01-01').date()
DEV_END=pd.Timestamp('2025-11-09').date()


def latest_price(area):
    obj=source.payload(Path('data/multiarea/energycharts')/f'price_{area}_2026_to_1109.json.gz')
    frame=pd.DataFrame([dict(timestamp=x['timestamp'],**x['values']) for x in obj['data']])
    return source.hourly(frame,['day_ahead_price'],'mean','price').day_ahead_price


def price_pairs():
    old=pd.read_csv('results/decade/processed.csv.gz',parse_dates=['HourUTC'])
    old=old[old.year.between(2019,2025)]
    zones={}
    for area in ['DK1','DK2']:
        past=old[old.PriceArea.eq(area)].set_index('HourUTC').SpotPriceEUR
        zones[area]=pd.concat([past,latest_price(area)]).sort_index()
    for area in ['DE-LU','NL','SE4']:
        zones[area]=pd.concat([source.chart('price',area,['day_ahead_price']).day_ahead_price,latest_price(area)]).sort_index()
    out=[]
    for pair in prior.PAIRS:
        a,b=pair.split('--')
        if zones[a].index.duplicated().any() or zones[b].index.duplicated().any():raise ValueError('Duplicate price timestamp')
        x=pd.concat([zones[a].rename('price_dk'),zones[b].rename('price_neighbor')],axis=1,join='outer').sort_index()
        expected=pd.date_range(x.index.min(),x.index.max(),freq='h')
        if not x.index.equals(expected) or x.isna().any().any():raise ValueError(f'Missing hourly price in {pair}')
        x.index.name='HourUTC';x=x.reset_index();x['pair']=pair
        x['year']=x.HourUTC.dt.tz_convert('Europe/Copenhagen').dt.year
        out.append(x)
    return pd.concat(out,ignore_index=True)


def weather_all(zone):
    old=prior.load_weather(zone)
    path=Path('data/external/weather')/f'{zone}_jma_day2_2026_to_1109.json.gz'
    with gzip.open(path,'rt',encoding='utf8') as f:j=json.loads(json.load(f)['body'])
    h=j['hourly']; new=pd.DataFrame({v:h[v+'_previous_day2'] for v in prior.WVARS},index=pd.to_datetime(h['time'],utc=True))
    new=new[new.index>=pd.Timestamp('2026-01-01T00:00Z')]
    data=pd.concat([old,new]).sort_index()
    if data.index.has_duplicates: raise ValueError('Duplicate weather timestamps')
    return data


def make_model(family):
    if family=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1200,random_state=42))
    return HistGradientBoostingClassifier(max_iter=120,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=100,l2_regularization=1,early_stopping=False,random_state=42)


def fit_all(train,target):
    sizes=train.groupby('pair').size().to_dict()
    weights=train.pair.map({p:len(train)/(len(prior.PAIRS)*n) for p,n in sizes.items()}).to_numpy()
    models={};rates=train.groupby('pair')[target].mean().to_dict()
    for family in ['logistic','boosting']:
        for name,cols in prior.FEATURES.items():
            print('Fitting',target,family,name,flush=True)
            clf=make_model(family)
            kw={'logisticregression__sample_weight':weights} if family=='logistic' else {'sample_weight':weights}
            clf.fit(train[cols],train[target],**kw);models[family+'_'+name]=clf
    return models,rates


def predictions(frame,models,rates):
    result={'pair_rate':frame.pair.map(rates).to_numpy()}
    for name,model in models.items():result[name]=model.predict_proba(frame[prior.FEATURES[name.split('_',1)[1]]])[:,1]
    return result


def day_interval(frame,delta,repeats=1000,block=7):
    dates=pd.date_range(TEST_START,TEST_END,freq='D').date
    daily=pd.DataFrame({'date':frame.date.to_numpy(),'delta':np.asarray(delta),'n':1}).groupby('date').sum().reindex(dates,fill_value=0).to_numpy()
    rng=np.random.default_rng(42);values=[]
    for _ in range(repeats):
        starts=rng.integers(0,len(dates)-block+1,int(np.ceil(len(dates)/block)))
        idx=np.concatenate([np.arange(s,s+block) for s in starts])[:len(dates)]
        d,n=daily[idx].sum(axis=0)
        if n: values.append(d/n)
    return np.quantile(values,[.025,.975]) if len(values)>=.8*repeats else (np.nan,np.nan)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=price_pairs();weather={z:weather_all(z) for z in ['DK1','DK2','DE-LU','NL','SE4']}
    with gzip.open('data/external/fred/brent_to_20261109.csv.gz','rt',encoding='utf8') as f: body=json.load(f)['body']
    oil=pd.read_csv(io.StringIO(body),parse_dates=['observation_date'],na_values=['.']).set_index('observation_date').DCOILBRENTEU.dropna().sort_index()
    data=prior.build_frame(raw,weather,oil)
    features=list(dict.fromkeys(sum(prior.FEATURES.values(),[])))
    data['complete']=np.isfinite(data[features]).all(axis=1)
    counts=data.groupby(['pair','year']).agg(hours=('complete','size'),complete=('complete','sum')).reset_index()
    counts.to_csv(OUT/'input_coverage.csv',index=False)
    data=data[data.complete].copy()
    predev=data[data.date<DEV_START];dev=data[data.date.between(DEV_START,DEV_END)]
    train=data[data.date<TEST_START];test=data[data.date.between(TEST_START,TEST_END)]
    if len(pd.date_range(TEST_START,TEST_END,freq='D'))!=365:raise ValueError('Test is not 365 days')
    expected=pd.date_range(pd.Timestamp(TEST_START,tz='Europe/Copenhagen').tz_convert('UTC'),
                           (pd.Timestamp(TEST_END,tz='Europe/Copenhagen')+pd.DateOffset(days=1)).tz_convert('UTC'),freq='h',inclusive='left')
    for pair,g in test.groupby('pair'):
        if not pd.DatetimeIndex(g.HourUTC).equals(expected): raise ValueError(f'Incomplete test hours for {pair}')
    if len(test.pair.unique())!=3:raise ValueError('Missing test pair')
    rows=[];devrows=[];preds=[];intervals=[];saved={};selections={}
    for target in ['any_negative','discordant']:
        print('Development fit:',target,flush=True)
        devmodels,devrates=fit_all(predev,target);dp=predictions(dev,devmodels,devrates)
        for (year,pair),g in dev.groupby(['year','pair']):
            ix=dev.index.get_indexer(g.index)
            for name,values in dp.items():devrows.append(dict(target=target,year=int(year),pair=pair,model=name,**prior.score(g[target].to_numpy(),values[ix])))
        ranks=pd.DataFrame(devrows);ranked=ranks[ranks.target.eq(target)].groupby('model').brier.mean().sort_values()
        selected=str(ranked.index[0]);selections[target]={'model':selected,'development_mean_brier':float(ranked.iloc[0])}
        print('Final fit:',target,'selected:',selected,flush=True)
        models,rates=fit_all(train,target);saved[target]={'models':models,'rates':rates,'selected':selected}
        pp=predictions(test,models,rates)
        for pair,g in test.groupby('pair'):
            ix=test.index.get_indexer(g.index);y=g[target].to_numpy()
            for name,allp in pp.items():
                p=allp[ix];rows.append(dict(target=target,pair=pair,model=name,**prior.score(y,p)))
                preds.append(pd.DataFrame({'HourUTC':g.HourUTC,'pair':pair,'target':target,'model':name,'y':y,'p':p}))
            for name,base in [('boosting_weather','boosting_price_history'),('boosting_weather_oil','boosting_weather'),
                              ('logistic_weather','logistic_price_history'),('logistic_weather_oil','logistic_weather'),(selected,'pair_rate')]:
                delta=(pp[base][ix]-y)**2-(pp[name][ix]-y)**2;lo,hi=day_interval(g,delta)
                intervals.append(dict(target=target,pair=pair,model=name,baseline=base,delta_brier=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),block_days=7))
    pd.DataFrame(devrows).to_csv(OUT/'development_metrics.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'test_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'paired_brier_intervals.csv',index=False)
    predictions_df=pd.concat(preds,ignore_index=True)
    predictions_df.to_csv(OUT/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    predictions_df['local_year']=pd.to_datetime(predictions_df.HourUTC,utc=True).dt.tz_convert('Europe/Copenhagen').dt.year
    byyear=[dict(target=t,pair=p,model=m,local_year=int(y),**prior.score(g.y.to_numpy(),g.p.to_numpy()))
            for (t,p,m,y),g in predictions_df.groupby(['target','pair','model','local_year'])]
    pd.DataFrame(byyear).to_csv(OUT/'test_by_calendar_year.csv',index=False)
    predictions_df['bin']=pd.cut(predictions_df.p,np.linspace(0,1,6),include_lowest=True)
    predictions_df.groupby(['target','pair','model','bin'],observed=True).agg(n=('y','size'),events=('y','sum'),observed=('y','mean'),mean_probability=('p','mean')).reset_index().to_csv(OUT/'calibration.csv',index=False)
    joblib.dump(saved,OUT/'models.joblib',compress=3)
    protocol=dict(data_through='2026-11-09',train_start=str(min(train.date)),train_end='2025-11-09',
        training_hours_by_pair=train.groupby('pair').size().to_dict(),development='2024-01-01 through 2025-11-09 (fit through 2023)',
        test_start=str(TEST_START),test_end=str(TEST_END),test_days=365,cutoff='previous local day 10:00',
        targets=['any_negative','discordant'],selections=selections,features=prior.FEATURES,
        assumptions=['weather 48-hour lead; run publication time not individually verified','Brent 14-day observation lag; historical publication/vintage unverified',
                     'previous study inspected 2024 and 2025, including the first 52 days of test'],seed=42)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
