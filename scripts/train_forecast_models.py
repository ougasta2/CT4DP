"""Exploratory all-hour classification using archived weather and lagged Brent."""
import gzip
import io
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_multiarea import load_data,block_interval,PAIRS

OUT=Path('results/forecast_models'); ROOT=Path('data/external')
WVARS=['temperature_2m','wind_speed_10m','cloud_cover']
CAL=['hour_sin','hour_cos','month_sin','month_cos','weekday_sin','weekday_cos','pair_de','pair_nl','pair_se']
LAGS=['dk_price_48h','dk_price_168h','neighbor_price_48h','neighbor_price_168h']
WEATHER=[f'{side}_{v}' for side in ['dk','neighbor'] for v in WVARS]
OIL=['brent_14d','brent_change_7d']
FEATURES={'calendar':CAL,'price_history':CAL+LAGS,'weather':CAL+LAGS+WEATHER,'weather_oil':CAL+LAGS+WEATHER+OIL}


def load_weather(zone):
    with gzip.open(ROOT/'weather'/f'{zone}_jma_day2.json.gz','rt',encoding='utf8') as f: obj=json.loads(json.load(f)['body'])
    h=obj['hourly']; t=pd.to_datetime(h['time'],utc=True)
    data=pd.DataFrame({v:h[v+'_previous_day2'] for v in WVARS},index=t)
    if data.index.has_duplicates: raise ValueError('Duplicate weather times')
    return data


def build_frame(raw,weather,brent):
    frames=[]
    for pair in PAIRS:
        dk,neighbor=pair.split('--')
        f=raw[raw.pair.eq(pair)].sort_values('HourUTC').copy()
        if f.HourUTC.duplicated().any() or not f.HourUTC.diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError('Price history must be a complete UTC hourly grid')
        for prefix,col in [('dk','price_dk'),('neighbor','price_neighbor')]:
            for lag in (48,168): f[f'{prefix}_price_{lag}h']=f[col].shift(lag)
        for side,area in [('dk',dk),('neighbor',neighbor)]:
            w=weather[area].reindex(pd.DatetimeIndex(f.HourUTC))
            for name in WVARS: f[f'{side}_{name}']=w[name].to_numpy()
        local=f.HourUTC.dt.tz_convert('Europe/Copenhagen')
        f['date']=local.dt.date
        for lag in (48,168):
            lag_day=(f.HourUTC-pd.Timedelta(hours=lag)).dt.tz_convert('Europe/Copenhagen').dt.date
            if not (lag_day < f.date).all(): raise ValueError('Price lag reaches the target delivery day')
        for name,val,period in [('hour',local.dt.hour,24),('month',local.dt.month-1,12),('weekday',local.dt.weekday,7)]:
            f[name+'_sin']=np.sin(2*np.pi*val/period);f[name+'_cos']=np.cos(2*np.pi*val/period)
        for name,target in [('de','DK1--DE-LU'),('nl','DK1--NL'),('se','DK2--SE4')]: f['pair_'+name]=(f.pair==target).astype(int)
        # This lag is a conservative proxy, not proof of historical publication vintage.
        asof=pd.DatetimeIndex(pd.to_datetime(f.date))-pd.Timedelta(days=14)
        f['brent_14d']=brent.reindex(asof,method='ffill').to_numpy()
        f['brent_change_7d']=f.brent_14d-brent.reindex(asof-pd.Timedelta(days=7),method='ffill').to_numpy()
        a,b=f.price_dk.lt(0),f.price_neighbor.lt(0)
        f['any_negative']=(a|b).astype(int);f['discordant']=(a^b).astype(int)
        # Latest fixed 48h lead always precedes previous-day 10:00 local for this horizon.
        assumed_issue=f.HourUTC-pd.Timedelta(hours=48)
        cutoff=(pd.to_datetime(f.date)-pd.Timedelta(days=1)+pd.Timedelta(hours=10)).dt.tz_localize('Europe/Copenhagen',ambiguous='raise',nonexistent='raise').dt.tz_convert('UTC')
        if not (assumed_issue.to_numpy() <= cutoff.to_numpy()).all(): raise ValueError('Weather lead violates cutoff')
        frames.append(f)
    return pd.concat(frames,ignore_index=True)


def score(y,p):
    return dict(n=len(y),events=int(y.sum()),prevalence=float(y.mean()),
                brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),
                average_precision=float(average_precision_score(y,p)) if y.sum() else np.nan,
                auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                mean_probability=float(p.mean()))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=load_data();weather={p:load_weather(p) for p in ['DK1','DK2','DE-LU','NL','SE4']}
    with gzip.open(ROOT/'fred'/'brent.csv.gz','rt',encoding='utf8') as f: body=json.load(f)['body']
    c=pd.read_csv(io.StringIO(body),parse_dates=['observation_date'],na_values=['.'])
    brent=c.set_index('observation_date').DCOILBRENTEU.dropna().sort_index()
    if brent.index.duplicated().any(): raise ValueError('Duplicate oil observation dates')
    data=build_frame(raw,weather,brent)
    required=list(dict.fromkeys(sum(FEATURES.values(),[])))
    data['input_complete']=np.isfinite(data[required]).all(axis=1)
    quality=data.groupby(['pair','year']).agg(total=('input_complete','size'),complete=('input_complete','sum')).reset_index()
    quality.to_csv(OUT/'input_coverage.csv',index=False)
    data=data[data.input_complete].copy()
    train=data[data.year.between(2019,2023)]; test=data[data.year.isin([2024,2025])]
    if train.empty or test.empty: raise ValueError('Missing train or evaluation data')
    models={}; predictions=[]; rows=[]
    for target in ['any_negative','discordant']:
        rates=train.groupby('pair')[target].mean().to_dict()
        tests=test.copy();tests['target']=target;tests['y']=tests[target]
        probabilities={'pair_rate':tests.pair.map(rates).to_numpy()}
        for family in ['logistic','boosting']:
            for feature_name,cols in FEATURES.items():
                print('Fitting',target,family,feature_name,flush=True)
                model=(make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1200,random_state=42)) if family=='logistic'
                       else HistGradientBoostingClassifier(max_iter=120,learning_rate=.05,max_leaf_nodes=15,
                           min_samples_leaf=100,l2_regularization=1,early_stopping=False,random_state=42))
                # Equal total pair weight; same rows and preprocessing in each ablation.
                sizes=train.groupby('pair').size().to_dict(); weights=train.pair.map({p:len(train)/(len(PAIRS)*n) for p,n in sizes.items()}).to_numpy()
                fitargs={'logisticregression__sample_weight':weights} if family=='logistic' else {'sample_weight':weights}
                model.fit(train[cols],train[target],**fitargs)
                name=family+'_'+feature_name; probabilities[name]=model.predict_proba(tests[cols])[:,1]
                models[(target,name)]=model
        for (year,pair),g in tests.groupby(['year','pair'],sort=True):
            idx=tests.index.get_indexer(g.index); y=g.y.to_numpy()
            for name,allp in probabilities.items():
                p=allp[idx];rows.append(dict(target=target,year=int(year),pair=pair,model=name,**score(y,p)))
                predictions.append(pd.DataFrame({'HourUTC':g.HourUTC,'year':year,'pair':pair,'target':target,'model':name,'y':y,'p':p}))
    scored=pd.DataFrame(rows);scored.to_csv(OUT/'metrics.csv',index=False)
    pred=pd.concat(predictions,ignore_index=True);pred.to_csv(OUT/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    intervals=[]
    for (target,year,pair),g in pred.groupby(['target','year','pair']):
        pivot=g.pivot(index='HourUTC',columns='model',values='p'); y=g[g.model=='pair_rate'].set_index('HourUTC').y.reindex(pivot.index).to_numpy()
        times=pd.to_datetime(pivot.index,utc=True).tz_convert('Europe/Copenhagen')
        f=pd.DataFrame({'year':int(year),'date':times.date})
        for family in ['logistic','boosting']:
            for a,b in [('price_history','weather'),('weather','weather_oil')]:
                x=pivot[family+'_'+a].to_numpy();z=pivot[family+'_'+b].to_numpy()
                delta=(x-y)**2-(z-y)**2;lo,hi=block_interval(f,delta,repeats=1000)
                intervals.append(dict(target=target,year=int(year),pair=pair,family=family,comparison=a+'_minus_'+b,
                                      brier_improvement=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),block_days=7))
    pd.DataFrame(intervals).to_csv(OUT/'ablation_intervals.csv',index=False)
    joblib.dump(models,OUT/'models.joblib',compress=3)
    selection={}
    for target in ['any_negative','discordant']:
        ranked=scored[(scored.year==2024)&(scored.target==target)].groupby('model').brier.mean().sort_values()
        selection[target]=dict(model=str(ranked.index[0]),mean_brier=float(ranked.iloc[0]))
    (OUT/'protocol.json').write_text(json.dumps(dict(train='2019-2023',development='2024',test='2025',
        cutoff='10:00 Europe/Copenhagen on day before delivery',weather='JMA GSM previous_day2, fixed 48h lead; issuance metadata not individually audited',
        oil='FRED/EIA Brent observation <= delivery date minus 14 days; historical publication vintage unverified',
        targets=['any_negative','discordant'],eligibility='all complete hours',features=FEATURES,development_selection=selection,
        prior_exposure='2024 and 2025 examined previously; exploratory evaluation',seed=42),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
