"""Conditional quantile boosting on past out-of-fold HGB residuals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_nonlinear as nonlinear
from train_dk1_conditional import choose

OUT=Path('results/dk1_demand_price')
Q=np.array([.1,.25,.5,.75,.9]);S=np.linspace(.1,.9,81)


def run():
    paths=['nonlinear_predictions.csv.gz','evaluation_2026_predictions.csv.gz']
    p=pd.concat([pd.read_csv(OUT/path,parse_dates=['HourUTC','day']).set_index('HourUTC') for path in paths]).sort_index()
    x=nonlinear.design(p);x['hgb_prediction']=p.hgb_price
    residual=p.DK1-p.hgb_price;outputs=[]
    for month in pd.date_range('2024-01-01','2026-11-01',freq='MS',inclusive='left'):
        tr=p.day.lt(month)&p.day.ge(month-pd.Timedelta(days=1095));ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        z=p.loc[ev,['day','hour','DK1','hgb_price']].copy();qs=[]
        for q in Q:
            model=HistGradientBoostingRegressor(loss='quantile',quantile=float(q),max_iter=60,learning_rate=.04,
                max_leaf_nodes=11,min_samples_leaf=100,l2_regularization=5.,early_stopping=False,random_state=42)
            model.fit(x.loc[tr],residual.loc[tr]);qs.append(model.predict(x.loc[ev]))
        quantiles=np.sort(np.stack(qs,axis=1),axis=1)
        samples=np.array([np.interp(S,Q,row) for row in quantiles])+z.hgb_price.to_numpy()[:,None]
        z['quantile_median']=samples[:,len(S)//2]
        z['quantile_band_opt']=[choose(vals,ref) for vals,ref in zip(samples,z.hgb_price)]
        z['q10']=samples[:,0];z['q90']=samples[:,-1]
        outputs.append(z);print('quantile',month.date(),flush=True)
    out=pd.concat(outputs).sort_index();out.to_csv(OUT/'quantile_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for year,g in out.groupby(out.day.dt.year):
        for name in ['hgb_price','quantile_median','quantile_band_opt']:
            rows.append({'year':year,'model':name,**base.metric(g.DK1,g[name])})
    table=pd.DataFrame(rows);table.to_csv(OUT/'quantile_metrics.csv',index=False)
    coverage=out.groupby(out.day.dt.year).apply(lambda g:pd.Series({'n':len(g),'coverage80':((g.DK1>=g.q10)&(g.DK1<=g.q90)).mean(),
       'width80':(g.q90-g.q10).mean()}));coverage.to_csv(OUT/'quantile_coverage.csv')
    (OUT/'quantile_protocol.json').write_text(json.dumps({'method':'conditional HGB quantile regressors of prior out-of-fold residuals; sorted quantiles; truncated 10–90% grid',
      'selection':'2024 only; compare fixed candidate to HGB baseline, MAE guardrail','later_periods':'2025 and 2026 observed in prior research; no independent claim',
      'limitations':['quantile grid omits distribution tails; this is not calibrated full predictive distribution','weather, settlement and fuel vintage not audited']},indent=2)+'\n',encoding='utf8')
    print(table.to_string(index=False),flush=True);print(coverage.to_string(),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
