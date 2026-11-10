"""Compare local, pooled and partially pooled retrospective pair models."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from energy_pilot.mining import apriori,mask_for

ROOT=Path('data/multiarea'); OUT=Path('results/multiarea_models')
PAIRS=['DK1--DE-LU','DK1--NL','DK2--SE4']
CAL=['hour_sin','hour_cos','month_sin','month_cos','weekend']
ENERGY=['common_wind','common_solar','common_load','diff_wind','diff_solar','diff_load']
COLS=CAL+ENERGY


def load_data():
    parts=[]
    for pair in PAIRS:
        a,b=pair.split('--')
        f=pd.read_csv(ROOT/f'paired_{a}_{b}_2019_2025.csv.gz',parse_dates=['HourUTC'])
        if f.HourUTC.duplicated().any() or not f.pair.eq(pair).all(): raise ValueError('Invalid pair-hour keys')
        parts.append(f[f.usable_features].copy())
    return pd.concat(parts,ignore_index=True)


class FeatureBuilder:
    def fit(self,frame):
        self.medians={}
        for p,g in frame.groupby('pair'):
            self.medians[p]=(float(g.load_dk_mwh.median()),float(g.load_neighbor.median()))
        return self

    def transform(self,frame):
        if set(frame.pair)-set(self.medians): raise ValueError('Unseen pair in feature transform')
        x=frame.copy(); local=x.HourUTC.dt.tz_convert('Europe/Copenhagen')
        for f,period,vals in [('hour',24,local.dt.hour),('month',12,local.dt.month-1)]:
            x[f+'_sin']=np.sin(2*np.pi*vals/period); x[f+'_cos']=np.cos(2*np.pi*vals/period)
        x['weekend']=(local.dt.weekday>=5).astype(int)
        x['date']=local.dt.date
        x['season']=np.select([local.dt.month.isin([12,1,2]),local.dt.month.isin([3,4,5]),local.dt.month.isin([6,7,8])],['winter','spring','summer'],default='autumn')
        x['time']=pd.cut(local.dt.hour,[-1,5,11,17,23],labels=['night','morning','afternoon','evening']).astype(str)
        m1=x.pair.map({k:v[0] for k,v in self.medians.items()});m2=x.pair.map({k:v[1] for k,v in self.medians.items()})
        for name,a,b in [('wind',x.wind_dk_mwh/x.load_dk_mwh,x.wind_neighbor/x.load_neighbor),
                          ('solar',x.solar_dk_mwh/x.load_dk_mwh,x.solar_neighbor/x.load_neighbor),
                          ('load',x.load_dk_mwh/m1,x.load_neighbor/m2)]:
            x['common_'+name]=(a+b)/2; x['diff_'+name]=a-b
        if not np.isfinite(x[COLS]).all().all(): raise ValueError('Invalid features')
        x['eligible']=x.price_dk.lt(0)|x.price_neighbor.lt(0)
        x['y']=(x.price_dk.lt(0)^x.price_neighbor.lt(0)).astype(int)
        return x


def expanded(values,pair,mode):
    if mode=='pooled': return values
    indicators=np.column_stack([(pair==p).astype(float) for p in PAIRS])
    # Half-sized pair deviations impose stronger L2 shrinkage than shared effects.
    return np.column_stack([values,indicators*.5,*[values*indicators[:,i,None]*.5 for i in range(len(PAIRS))]])


def fit_models(train):
    y=train.y.to_numpy(); x=train[COLS].to_numpy(); scaler=StandardScaler().fit(x)
    z=scaler.transform(x); pair=train.pair.to_numpy()
    counts=train.groupby('pair').size().to_dict()
    weights=np.array([len(train)/(len(PAIRS)*counts[p]) for p in pair])
    models={}
    for mode in ['pooled','partial_pool']:
        model=LogisticRegression(C=1.,max_iter=3000,random_state=42).fit(expanded(z,pair,mode),y,sample_weight=weights)
        models[mode]=(scaler,model)
    local={}; calendar={}
    for p in PAIRS:
        keep=pair==p; ss=StandardScaler().fit(x[keep]); clf=LogisticRegression(C=1.,max_iter=3000,random_state=42).fit(ss.transform(x[keep]),y[keep]); local[p]=(ss,clf)
        c=StandardScaler().fit(x[keep,:len(CAL)]); model=LogisticRegression(C=1.,max_iter=3000,random_state=42).fit(c.transform(x[keep,:len(CAL)]),y[keep]); calendar[p]=(c,model)
    transfers={}
    for target in PAIRS:
        for scope in ['loo_pair','loo_disjoint']:
            target_areas=set(target.split('--'))
            sources=[p for p in PAIRS if p!=target and (scope=='loo_pair' or not target_areas.intersection(p.split('--')))]
            keep=np.isin(pair,sources); xx=x[keep]; yy=y[keep]; source_pairs=pair[keep]
            ss=StandardScaler().fit(xx); sizes=pd.Series(source_pairs).value_counts().to_dict()
            w=np.array([len(xx)/(len(sources)*sizes[p]) for p in source_pairs])
            clf=LogisticRegression(C=1.,max_iter=3000,random_state=42).fit(ss.transform(xx),yy,sample_weight=w)
            transfers[(target,scope)]=(ss,clf,sources)
    return models,local,calendar,transfers,train.groupby('pair').y.mean().to_dict()


def predict(test,models,local,calendar,transfers,rates):
    pair=test.pair.to_numpy();x=test[COLS].to_numpy()
    out={'rate':test.pair.map(rates).to_numpy()}
    out['local']=np.empty(len(test))
    for p,(ss,model) in local.items():
        keep=pair==p;out['local'][keep]=model.predict_proba(ss.transform(x[keep]))[:,1]
    out['calendar']=np.empty(len(test))
    for p,(ss,model) in calendar.items():
        keep=pair==p;out['calendar'][keep]=model.predict_proba(ss.transform(x[keep,:len(CAL)]))[:,1]
    for scope in ['loo_pair','loo_disjoint']:
        out[scope]=np.empty(len(test))
        for p in PAIRS:
            keep=pair==p;ss,model,_=transfers[(p,scope)]
            out[scope][keep]=model.predict_proba(ss.transform(x[keep]))[:,1]
    for mode,(ss,model) in models.items():
        out[mode]=model.predict_proba(expanded(ss.transform(x),pair,mode))[:,1]
    return out


def metrics(y,p):
    return dict(n=len(y),events=int(y.sum()),prevalence=float(y.mean()),brier=float(brier_score_loss(y,p)),
                log_loss=float(log_loss(y,p,labels=[0,1])),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                average_precision=float(average_precision_score(y,p)) if y.sum() else None,mean_probability=float(np.mean(p)))


def block_interval(frame,delta,repeats=1000,block=7):
    year=int(frame.year.iloc[0]); grid=pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D').date
    counts=pd.DataFrame({'date':frame.date.to_numpy(),'delta':np.asarray(delta),'n':1}).groupby('date').sum().reindex(grid,fill_value=0).to_numpy()
    rng=np.random.default_rng(42);vals=[]
    for _ in range(repeats):
        starts=rng.integers(0,len(grid)-block+1,int(np.ceil(len(grid)/block)))
        chosen=np.concatenate([np.arange(s,s+block) for s in starts])[:len(grid)]
        d,n=counts[chosen].sum(axis=0)
        if n: vals.append(d/n)
    return np.quantile(vals,[.025,.975]) if len(vals)>=.8*repeats else (np.nan,np.nan)


def macro_block(test,probabilities,baseline,model,repeats=1000,block=7):
    """Synchronous day blocks across all pairs; equal weight for each pair."""
    year=int(test.year.iloc[0]);grid=pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D').date
    y=test.y.to_numpy();delta=(probabilities[baseline]-y)**2-(probabilities[model]-y)**2
    tmp=pd.DataFrame({'date':test.date.to_numpy(),'pair':test.pair.to_numpy(),'delta':delta,'n':1})
    daily=tmp.groupby(['date','pair'])[['delta','n']].sum().unstack('pair').reindex(grid,fill_value=0).fillna(0)
    values=daily.to_numpy(); rng=np.random.default_rng(42); estimates=[]
    for _ in range(repeats):
        starts=rng.integers(0,len(grid)-block+1,int(np.ceil(len(grid)/block)))
        ix=np.concatenate([np.arange(s,s+block) for s in starts])[:len(grid)]
        totals=pd.Series(values[ix].sum(axis=0),index=daily.columns)
        n=totals['n']; d=totals['delta']
        if (n>0).all(): estimates.append(float((d/n).mean()))
    return np.quantile(estimates,[.025,.975]) if len(estimates)>=.8*repeats else (np.nan,np.nan)


def baskets(frame):
    return [frozenset([f'cluster=C{c}',f'season={s}',f'time={t}']) for c,s,t in zip(frame.cluster,frame.season,frame.time)]


def fit_rules(all_train,eligible):
    ss=StandardScaler().fit(all_train[ENERGY]); km=KMeans(n_clusters=4,n_init=20,random_state=42).fit(ss.transform(all_train[ENERGY]))
    eligible=eligible.copy();eligible['cluster']=km.predict(ss.transform(eligible[ENERGY]))
    selected={}
    for p,g in eligible.groupby('pair'):
        bs=baskets(g); y=g.y.to_numpy();options=[]
        for ant,n in apriori(bs,min_support=.05,max_size=3).items():
            if not any(v.startswith('cluster=') for v in ant): continue
            mask=mask_for(bs,ant);e=int(y[mask].sum())
            if n>=40 and e>=20 and y.mean()>0: options.append((e/n/y.mean(),n,tuple(sorted(ant)),e))
        if options: selected[p]=max(options)
    return ss,km,selected


def run():
    OUT.mkdir(parents=True,exist_ok=True); all_data=load_data()
    metrics_rows=[]; preds=[];intervals=[]; macro=[]; rules=[]; fitted_final=None
    # 2023 backtest; 2024 development and 2025 exploratory evaluation share the same frozen fit.
    for train_end,years in [(2022,[2023]),(2023,[2024,2025])]:
        print(f'Fitting through {train_end}, evaluating {years}',flush=True)
        raw_train=all_data[all_data.year.between(2019,train_end)]
        builder=FeatureBuilder().fit(raw_train)
        transformed=builder.transform(raw_train); train=transformed[transformed.eligible].copy()
        models,local,calendar,transfers,rates=fit_models(train)
        cs,km,selected=fit_rules(transformed,train)
        if train_end==2023:
            fitted_final=dict(load_medians=builder.medians,models=models,local=local,calendar=calendar,transfers=transfers,rates=rates,cluster_scaler=cs,kmeans=km,selected_rules=selected)
            centers=pd.DataFrame(cs.inverse_transform(km.cluster_centers_),columns=ENERGY);centers.index.name='cluster';centers.to_csv(OUT/'cluster_centers.csv')
        for year in years:
            frame=builder.transform(all_data[all_data.year==year]); test=frame[frame.eligible].copy(); probabilities=predict(test,models,local,calendar,transfers,rates)
            test['cluster']=km.predict(cs.transform(test[ENERGY])); y=test.y.to_numpy()
            for baseline in ['rate','calendar','local']:
                lo,hi=macro_block(test,probabilities,baseline,'pooled')
                means=[np.mean((probabilities[baseline][test.pair.eq(p)]-y[test.pair.eq(p)])**2-
                               (probabilities['pooled'][test.pair.eq(p)]-y[test.pair.eq(p)])**2) for p in PAIRS]
                macro.append(dict(year=year,baseline=baseline,model='pooled',delta_brier=float(np.mean(means)),ci_low=float(lo),ci_high=float(hi),block_days=7,repeats=1000))
            for p in PAIRS:
                keep=test.pair.eq(p).to_numpy(); t=test.iloc[np.flatnonzero(keep)];yt=y[keep]
                for name,vals in probabilities.items():
                    pp=vals[keep];metrics_rows.append(dict(train_end=train_end,year=year,pair=p,model=name,**metrics(yt,pp)))
                    preds.append(pd.DataFrame({'HourUTC':t.HourUTC,'year':year,'pair':p,'model':name,'y':yt,'probability':pp}))
                for name in ['pooled','partial_pool','loo_pair','loo_disjoint']:
                    for baseline in ['rate','calendar','local']:
                        delta=(probabilities[baseline][keep]-yt)**2-(probabilities[name][keep]-yt)**2
                        lo,hi=block_interval(t,delta)
                        intervals.append(dict(year=year,pair=p,model=name,baseline=baseline,delta_brier=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),block_days=7,repeats=1000))
                if p in selected:
                    lift,n,ant,events=selected[p]; mask=mask_for(baskets(t),frozenset(ant));matches=int(mask.sum());e=int(yt[mask].sum());base=yt.mean()
                    rules.append(dict(year=year,pair=p,antecedent=' & '.join(ant),train_n=n,train_events=events,train_lift=lift,
                                      test_n=matches,test_events=e,test_base=base,test_lift=(e/matches/base if matches and base else np.nan)))
    table=pd.DataFrame(metrics_rows); table.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'paired_brier_intervals.csv',index=False)
    pd.DataFrame(rules).to_csv(OUT/'rules.csv',index=False)
    pred=pd.concat(preds,ignore_index=True)
    pred.to_csv(OUT/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(macro).to_csv(OUT/'macro_brier_intervals.csv',index=False)
    jan_sep=pred[(pred.year==2025)&(pd.to_datetime(pred.HourUTC,utc=True).dt.tz_convert('Europe/Copenhagen').dt.month<=9)]
    sens=[dict(pair=p,model=m,**metrics(g.y.to_numpy(),g.probability.to_numpy())) for (p,m),g in jan_sep.groupby(['pair','model'])]
    pd.DataFrame(sens).to_csv(OUT/'sensitivity_2025_jan_sep.csv',index=False)
    pred['bin']=pd.cut(pred.probability,np.linspace(0,1,6),include_lowest=True)
    cal=pred.groupby(['year','pair','model','bin'],observed=True).agg(n=('y','size'),events=('y','sum'),observed_rate=('y','mean'),mean_probability=('probability','mean')).reset_index()
    cal.to_csv(OUT/'calibration.csv',index=False)
    # Rank using 2024 equal-pair mean Brier, including the constant baseline; no 2025 tuning.
    ranking=table[table.year.eq(2024)&table.model.isin(['rate','calendar','local','pooled','partial_pool'])].groupby('model').brier.mean().sort_values()
    protocol=dict(train='2019-2023',development='2024',test='2025',backtest_train='2019-2022',backtest_test='2023',
                  target='exactly one negative, conditional on at least one negative',feature_names=COLS,
                  weight='equal total weight per pair',C=1,partial_pool='L2 regularized pair effects scaled 0.5',
                  chosen_model=str(ranking.index[0]),development_mean_brier=ranking.to_dict(),
                  prior_exposure='2024 and 2025 previously explored; all evaluation is exploratory',seed=42)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    joblib.dump(fitted_final,OUT/'models.joblib',compress=3)


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
