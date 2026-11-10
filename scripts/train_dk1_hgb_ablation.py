"""Matched nonlinear feature ablation: own demand and stacked base prediction."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_nonlinear as nonlinear

OUT=Path('results/dk1_demand_price')


def run():
    p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC').sort_index()
    x=nonlinear.design(p)
    # Full model is recomputed along with every ablation on identical folds.
    variants={'full':x,'no_demand':x.drop(columns=['load_forecast_DK1','load_forecast_DK2']),
              'no_stacked_price':x.drop(columns=['base_prediction'])}
    outputs=[]
    for month in pd.date_range('2023-01-01','2026-01-01',freq='MS',inclusive='left'):
        train=p.day.lt(month)&p.day.ge(month-pd.Timedelta(days=1095))
        ev=p.day.ge(month)&p.day.lt(month+pd.offsets.MonthBegin(1))
        z=p.loc[ev,['day','hour','DK1']].copy()
        for name,xx in variants.items():
            model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,learning_rate=.04,max_leaf_nodes=15,
                min_samples_leaf=100,l2_regularization=5.,early_stopping=False,random_state=42)
            model.fit(xx.loc[train],p.DK1.loc[train]);z[name]=model.predict(xx.loc[ev])
        outputs.append(z)
        print('hgb ablation',month.date(),flush=True)
    pred=pd.concat(outputs).sort_index();pred.to_csv(OUT/'nonlinear_ablation_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for year,g in pred.groupby(pred.day.dt.year):
        for name in variants:rows.append({'year':year,'model':name,**base.metric(g.DK1,g[name])})
    table=pd.DataFrame(rows);table.to_csv(OUT/'nonlinear_ablation_metrics.csv',index=False)
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
