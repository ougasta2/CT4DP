"""EXAA early auction ablation at 11:00; 2022 plus May–Oct 2026.
EXAA historical publication times/revisions are not individually audited.
"""
import gzip,io,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_exaa_2022 as history
import train_dk1_nonlinear as nonlinear

OUT=Path('results/dk1_exaa_2026')


def exaa_2026():
    with gzip.open('data/external/exaa_entsoe/AT_2026.json.gz','rt',encoding='utf8') as f:obj=json.load(f)
    z=pd.read_csv(io.StringIO(obj['body']),parse_dates=['ts_utc'])
    if z.ts_utc.duplicated().any():raise ValueError('duplicate EXAA observations')
    # Unknown currency labels cannot safely be interpreted as EUR.
    z=z[z.currency.eq('EUR')].copy()
    z['hour']=z.ts_utc.dt.floor('h');z['minute']=z.ts_utc.dt.minute
    rows=z.groupby('hour').minute.agg(lambda a:tuple(sorted(a)))
    values=z.groupby('hour').eur_per_mwh.mean()
    return values.where(rows.map(lambda a:a==(0,15,30,45))).dropna()


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    past=pd.read_csv('results/dk1_demand_price/predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
    p26=pd.read_csv('results/dk1_demand_price/evaluation_2026_predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
    p=pd.concat([past[past.day.dt.year.eq(2022)],p26]).sort_index()
    x=nonlinear.design(p)
    s=pd.concat([history.early_prices(),exaa_2026()]).sort_index()
    if s.index.has_duplicates:raise ValueError('overlapping archive years')
    value=s.reindex(p.index)
    x['exaa_at']=value.to_numpy()
    x['exaa_minus_dk1_previous']=x.exaa_at-x.DK1_lag1
    # Baseline and treatment are both conditioned on price availability.
    good=x.notna().all(axis=1);p=p.loc[good].copy();x=x.loc[good].copy()
    variants={'without_exaa':x.drop(columns=['exaa_at','exaa_minus_dk1_previous']),'with_exaa':x}
    outputs=[]
    for month in pd.date_range('2026-05-01','2026-11-01',freq='MS',inclusive='left'):
        train=p.day.lt(month);test=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        if train.sum()<5000 or test.sum()==0:continue
        z=p.loc[test,['day','hour','DK1','lasso_own_demand_fuel','hgb_price']].copy();z['exaa_raw']=x.exaa_at.loc[test]
        for name,features in variants.items():
            m=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,learning_rate=.04,max_leaf_nodes=15,
                 min_samples_leaf=100,l2_regularization=5.,early_stopping=False,random_state=42)
            m.fit(features.loc[train],p.DK1.loc[train]);z[name]=m.predict(features.loc[test])
        outputs.append(z);print('EXAA 11:00',month.date(),'train',int(train.sum()),'eval',int(test.sum()),flush=True)
    result=pd.concat(outputs).sort_index();result.to_csv(OUT/'predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for name in ['lasso_own_demand_fuel','hgb_price','exaa_raw','without_exaa','with_exaa']:
        rows.append({'period':'2026-05-to-10','model':name,**base.metric(result.DK1,result[name])})
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'metrics.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps({'cutoff':'11:00 Copenhagen on D-1, before 12:00 market gate; not comparable to 10:00 without matching cutoff',
      'product':'AT early 10:15 EXAA auction only: APG for 2022 and ENTSO-E via entsoemcp for 2026',
      'training':'2022 plus available earlier 2026 months; no 2023–2025 EXAA early price present',
      'evaluation':'2026 May to Oct where all 4 EXAA quarter-hour values explicitly have EUR currency; same evaluation hours for all variants',
      'availability':'scheduled early auction results by 11:00 are assumed, historical exact release/revision timestamps not audited; live usage requires timestamp validation',
      'other_limits':['2026 partly studied before, not independent','weather, demand and fuel vintages unverified','15-minute DK1 prices aggregated hourly']},indent=2)+'\n',encoding='utf8')
    print(metrics.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
