"""11:00 DE EXAA residual/gating study. Chronological exploratory 2026 test."""
import gzip,io,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_nonlinear as nonlinear

OUT=Path('results/dk1_exaa_de')


def load_exaa(zone):
    with gzip.open(f'data/external/exaa_entsoe/{zone}_2026.json.gz','rt',encoding='utf8') as f:obj=json.load(f)
    a=pd.read_csv(io.StringIO(obj['body']),parse_dates=['ts_utc']);a=a[a.currency.eq('EUR')].copy()
    if a.ts_utc.duplicated().any():raise ValueError('duplicate EXAA timestamp')
    a['hour']=a.ts_utc.dt.floor('h');a['minute']=a.ts_utc.dt.minute
    count=a.groupby('hour').minute.agg(lambda s:tuple(sorted(s)))
    return a.groupby('hour').eur_per_mwh.mean().where(count.map(lambda s:s==(0,15,30,45))).dropna()


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv('results/dk1_demand_price/evaluation_2026_predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC').sort_index()
    x=nonlinear.design(p);x['EXAA_DE']=load_exaa('DE_LU').reindex(p.index);x['EXAA_AT']=load_exaa('AT').reindex(p.index)
    x['early_DE_AT_spread']=x.EXAA_DE-x.EXAA_AT;x['early_DE_DK1prev_spread']=x.EXAA_DE-x.DK1_lag1
    x['early_DE_hgb_spread']=x.EXAA_DE-p.hgb_price;x['hgb_price']=p.hgb_price
    ok=x.notna().all(axis=1);p=p.loc[ok].copy();x=x.loc[ok]
    pieces=[]
    for month in pd.date_range('2026-08-01','2026-11-01',freq='MS',inclusive='left'):
        tr=p.day.lt(month);ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        if tr.sum()<250 or ev.sum()==0:continue
        z=p.loc[ev,['day','hour','DK1','hgb_price']].copy();z['exaa_de']=x.EXAA_DE.loc[ev]
        residual=(p.DK1-x.EXAA_DE).loc[tr]
        model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=80,learning_rate=.04,max_leaf_nodes=7,
                 min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=42)
        model.fit(x.loc[tr],residual);z['de_residual_median']=z.exaa_de+model.predict(x.loc[ev])
        z['blend_75_de']=.75*z.exaa_de+.25*z.hgb_price
        # Classifier predicts whether DE itself falls in ±5%; true target used only on past data.
        yt=p.DK1.loc[tr];e=x.EXAA_DE.loc[tr];eligible=yt.abs()>=20
        label=(e-yt).abs()<=.05*yt.abs()
        clf=HistGradientBoostingClassifier(max_iter=80,learning_rate=.04,max_leaf_nodes=7,min_samples_leaf=40,
                                            l2_regularization=10,early_stopping=False,random_state=42)
        clf.fit(x.loc[tr].loc[eligible],label.loc[eligible]);prob=clf.predict_proba(x.loc[ev])[:,1]
        for threshold in [.4,.5,.6]:
            z[f'gate_{threshold}']=np.where(prob>=threshold,z.exaa_de,z.hgb_price)
        z['estimated_de_hit_probability']=prob
        pieces.append(z);print('DE early price',month.date(),'train',int(tr.sum()),'test',int(ev.sum()),flush=True)
    result=pd.concat(pieces).sort_index();result.to_csv(OUT/'predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for month,g in result.groupby(result.day.dt.strftime('%Y-%m')):
        for name in ['hgb_price','exaa_de','de_residual_median','blend_75_de','gate_0.4','gate_0.5','gate_0.6']:
            rows.append({'month':month,'model':name,**base.metric(g.DK1,g[name])})
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics.csv',index=False)
    # Candidate selected on August only, holding MAE no worse than raw DE.
    aug=table[table.month.eq('2026-08')];guard=aug.loc[aug.model.eq('exaa_de'),'mae'].iloc[0]
    ranked=aug[aug.mae.le(guard)].sort_values(['rate_5pct','mae'],ascending=[False,True])
    selected=ranked.iloc[0].model
    test=result[result.day.ge(pd.Timestamp('2026-09-01'))]
    combined=pd.DataFrame([{'period':'2026-09-to-10','model':name,**base.metric(test.DK1,test[name])}
        for name in dict.fromkeys(['hgb_price','exaa_de',selected]) if name in test])
    combined.to_csv(OUT/'test_metrics.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps({'selection':'August 2026; max hit rate with MAE no worse than raw DE EXAA','selected':selected,
      'test':'September–October 2026, exploratory since aggregate period already examined',
      'sampling':'only hours with all four EUR-labelled early-auction quarter prices for AT and DE',
      'training':'prior 2026 hours only; monthly refit; no same-day results used',
      'cutoff':'11:00 D-1 Copenhagen; normal 10:15 early auction assumed published by 11:00; no per-record release vintage',
      'caveats':['2026 is model development due to prior examination','original publication and revisions not verified','hourly averages'],
      'sources':['https://entsoemcp.com/licensing','https://www.exaa.at/en/energytrading/handel-mit-exaa/']},indent=2)+'\n',encoding='utf8')
    print('selected',selected,flush=True);print(table.to_string(index=False),flush=True);print(combined.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
