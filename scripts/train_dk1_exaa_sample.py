"""EXAA 11:00 point-in-time proxy: pre-specified sampled daily APG experiment.
APG sample is every fourth calendar day, not a full-year score. Annual backfill
lacks historical release timestamps; 11:00 timely publication is an assumption.
"""
import gzip,json
import argparse
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_nonlinear as nonlinear
from train_dk1_exaa_2026 import exaa_2026

ROOT=Path('data/external/apg_exaa_sample')
OUT=Path('results/dk1_exaa_sample')


def read_apg(path):
    with gzip.open(path,'rt',encoding='utf8') as f:obj=json.load(f)
    data=obj['body']['ResponseData'];names=[x['InternalName'] for x in data['ValueColumns']]
    position=names.index('EXAAD1Price');rows=data['ValueRows']
    day=pd.Timestamp(path.stem.split('.')[0]);local=[];values=[]
    for row in rows:
        t=row['TF'].replace('2A:','02:').replace('2B:','02:')
        local.append(pd.Timestamp(day.strftime('%Y-%m-%d')+' '+t))
        cell=row['V'][position];values.append(cell['V'] if not cell.get('M',False) else np.nan)
    ix=pd.DatetimeIndex(local).tz_localize('Europe/Vienna',ambiguous='infer',nonexistent='raise').tz_convert('UTC')
    if ix.has_duplicates or len(ix) not in [23,24,25]:raise ValueError('bad local day')
    return pd.Series(values,index=ix,dtype=float)


def source(full=False):
    paths=sorted(ROOT.glob('*.json.gz'))
    if not full:
        paths=[p for p in paths if pd.Timestamp(p.stem.split('.')[0]).toordinal()%4==0]
    expected_dates=pd.date_range('2023-01-01','2026-01-01',inclusive='left',freq='D')
    if not full:
        expected_dates=[day for day in expected_dates if day.toordinal()%4==0]
    expected_names={f'{day:%Y-%m-%d}.json.gz' for day in expected_dates}
    actual_names={path.name for path in paths}
    if actual_names != expected_names:
        raise ValueError(f'APG archive mismatch: {len(expected_names-actual_names)} missing and {len(actual_names-expected_names)} unexpected days')
    hist=pd.concat([read_apg(p) for p in paths]).sort_index()
    if hist.index.has_duplicates:raise ValueError('duplicate APG sample timestamps')
    out=pd.concat([hist,exaa_2026()]).sort_index()
    if out.index.has_duplicates:raise ValueError('overlap between APG and ENTSOE')
    return out


def paired_interval(frame, model, baseline='hgb_price', seed=42, repeats=2000):
    """Calendar-week bootstrap retains unsampled days as empty days."""
    y=frame.DK1.to_numpy();a=frame[baseline].to_numpy();b=frame[model].to_numpy()
    eligible=np.abs(y)>=20
    daily=pd.DataFrame({'day':frame.day.to_numpy(),'n':1,'eligible':eligible.astype(int),
        'mae_improvement':np.abs(a-y)-np.abs(b-y),
        'hit_improvement':((np.abs(b-y)<=.05*np.abs(y))&eligible).astype(int)-((np.abs(a-y)<=.05*np.abs(y))&eligible).astype(int)})
    dates=pd.date_range(frame.day.min(),frame.day.max(),freq='D')
    daily=daily.groupby('day').sum().reindex(dates,fill_value=0).to_numpy()
    rng=np.random.default_rng(seed);values=[];block=7
    for _ in range(repeats):
        starts=rng.integers(0,len(daily)-block+1,size=int(np.ceil(len(daily)/block)))
        idx=np.concatenate([np.arange(s,s+block) for s in starts])[:len(daily)]
        n,eligible_n,mae,hits=daily[idx].sum(axis=0)
        if n and eligible_n:values.append((mae/n,hits/eligible_n))
    return np.quantile(values,[.025,.975],axis=0)


def run(full=False):
    global OUT
    if full:OUT=Path('results/dk1_exaa_full')
    OUT.mkdir(parents=True,exist_ok=True)
    past=pd.read_csv('results/dk1_demand_price/nonlinear_predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
    fresh=pd.read_csv('results/dk1_demand_price/evaluation_2026_predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
    p=pd.concat([past,fresh]).sort_index();x=nonlinear.design(p)
    x['hgb_prediction']=p.hgb_price.to_numpy();x['exaa_at']=source(full).reindex(p.index).to_numpy()
    x['exaa_minus_dk1_previous']=x.exaa_at-x.DK1_lag1
    good=x.notna().all(axis=1);x=x.loc[good];p=p.loc[good].copy()
    variants={'stack_without_exaa':x.drop(columns=['exaa_at','exaa_minus_dk1_previous']),'stack_with_exaa':x}
    result=[]
    for month in pd.date_range('2024-01-01','2026-11-01',freq='MS',inclusive='left'):
        tr=p.day.lt(month);ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        if tr.sum()<1000 or ev.sum()==0:continue
        z=p.loc[ev,['day','hour','DK1','hgb_price']].copy();z['exaa_raw']=x.exaa_at.loc[ev]
        for frac in [.25,.5,.75]:z['blend_'+str(frac)]=(1-frac)*z.hgb_price+frac*z.exaa_raw
        for name,features in variants.items():
            m=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,learning_rate=.04,max_leaf_nodes=15,
                min_samples_leaf=40,l2_regularization=10.,early_stopping=False,random_state=42)
            m.fit(features.loc[tr],p.DK1.loc[tr]);z[name]=m.predict(features.loc[ev])
            if full and month==pd.Timestamp('2026-10-01') and name=='stack_with_exaa':
                artifact={'model':m,'feature_columns':features.columns.tolist(),
                          'fit_before':str(month.date()),'last_training_target':str(p.day.loc[tr].max().date()),
                          'issue':'11:00 D-1 Europe/Copenhagen; historical source vintages unaudited'}
                path=OUT/'october_2026_model.joblib'
                joblib.dump(artifact,path)
                restored=joblib.load(path)
                difference=np.max(np.abs(restored['model'].predict(features.loc[ev,restored['feature_columns']])-z[name].to_numpy()))
                if difference>1e-9:raise ValueError('Serialized model failed prediction round trip')
                print('serialized October model max difference',difference,flush=True)
        z['ridge_with_exaa']=make_pipeline(StandardScaler(),Ridge(alpha=100.)).fit(x.loc[tr],p.DK1.loc[tr]).predict(x.loc[ev])
        result.append(z);print('EXAA sampled',month.date(),'train',int(tr.sum()),'test',int(ev.sum()),flush=True)
    out=pd.concat(result).sort_index();out.to_csv(OUT/'predictions.csv.gz',compression={'method':'gzip','mtime':0})
    models=['hgb_price','exaa_raw','blend_0.25','blend_0.5','blend_0.75','stack_without_exaa','stack_with_exaa','ridge_with_exaa']
    rows=[]
    for year,g in out.groupby(out.day.dt.year):
        for name in models:rows.append({'year':year,'days':g.day.nunique(),'model':name,**base.metric(g.DK1,g[name])})
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics.csv',index=False)
    dev=table[table.year.eq(2024)].copy();mae=dev.loc[dev.model.eq('hgb_price'),'mae'].iloc[0]
    rank=dev[dev.mae.le(mae)].sort_values(['rate_5pct','mae'],ascending=[False,True]);selected=rank.iloc[0].model
    rank.to_csv(OUT/'development_ranking.csv',index=False)
    intervals=[]
    for year,g in out.groupby(out.day.dt.year):
        bounds=paired_interval(g,selected)
        intervals.append({'year':int(year),'model':selected,'baseline':'hgb_price',
                          'mae_improvement_low':bounds[0,0],'mae_improvement_high':bounds[1,0],
                          'hit_improvement_low':bounds[0,1],'hit_improvement_high':bounds[1,1],
                          'block_days':7})
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps({'cutoff':'11:00 D-1 Copenhagen; scheduled 10:15 EXAA price assumed known by cutoff, exact vintages not audited',
      'market':'Austria EXAA early auction (not market coupling); APG 2023–2025, ENTSO-E 2026 with EUR labels only',
      'sampling':('2023–2025 all calendar days; 2026 available EUR-labelled data May–Oct' if full else '2023–2025 fixed one in four calendar days selected by date.toordinal() % 4 == 0; 2026 available EUR-labelled data May–Oct'),
      'selection':('highest 2024 full-year ±5% rate' if full else 'highest 2024 sampled ±5% rate')+' subject to MAE <= 10am HGB baseline; 2025/2026 excluded from selection', 'selected':selected,
      'caveats':[('2026 conditional on source coverage' if full else 'conditional on sampled days and source coverage, not full-year'),'historical EXAA availability and revisions not individually audited','weather, fuel and load publication not individually verified','2025–2026 examined in earlier project work','15-minute DK1 prices averaged hourly']},indent=2)+'\n',encoding='utf8')
    print('selected:',selected,flush=True);print(table.to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=2):run(args.full)
