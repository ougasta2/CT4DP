"""Frozen linear previous-day cross-zone-spread corrections for DE early price."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train_dk1_preauction as base
from train_price_regression import local_lag
from train_dk1_exaa_sample import paired_interval

OUT=Path('results/dk1_exaa_de')
p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
prices=base.load_prices()
lags={}
for zone in ['DK1','DE-LU']:
 series=prices[zone]
 frame=pd.DataFrame({'HourUTC':series.index,'price':series.to_numpy()})
 lags[zone]=pd.Series(local_lag(frame,1),index=series.index).reindex(p.index)
p['prior_spread']=lags['DK1']-lags['DE-LU']
if not np.isfinite(p.prior_spread).all():
 raise ValueError('Missing previous-day prices for spread correction')
for v in [0,.1,.25,.5,.75,1.0]:p['lag_spread_'+str(v)]=p.exaa_de+v*p.prior_spread
models=['lag_spread_'+str(v) for v in [0,.1,.25,.5,.75,1.0]]
rows=[]
for month,g in p.groupby(p.day.dt.strftime('%Y-%m')):
 for model in models:rows.append({'month':month,'model':model,**base.metric(g.DK1,g[model])})
tab=pd.DataFrame(rows);tab.to_csv(OUT/'spread_metrics.csv',index=False)
dev=tab[tab.month.eq('2026-08')];guard=dev[dev.model.eq('lag_spread_0')].mae.iloc[0]
rank=dev[dev.mae.le(guard)].sort_values(['rate_5pct','mae'],ascending=[False,True]);selected=rank.iloc[0].model
test=p[p.day.ge('2026-09-01')]
combined=pd.DataFrame([{'period':'2026-09-to-10','model':name,**base.metric(test.DK1,test[name])} for name in dict.fromkeys(['lag_spread_0',selected])])
combined.to_csv(OUT/'spread_test_metrics.csv',index=False)
bounds=paired_interval(test,selected,baseline='lag_spread_0')
pd.DataFrame([{'period':'2026-09-to-10','model':selected,'baseline':'lag_spread_0',
 'mae_improvement_low':bounds[0,0],'mae_improvement_high':bounds[1,0],
 'hit_improvement_low':bounds[0,1],'hit_improvement_high':bounds[1,1],
 'block_days':7}]).to_csv(OUT/'spread_test_paired_interval.csv',index=False)
(OUT/'spread_protocol.json').write_text(json.dumps({'selection':'August 2026, +/-5% accuracy subject to MAE no worse than raw EXAA DE','selected':selected,
'test':'Sep–Oct 2026, exploratory since full period already viewed','cutoff':'11:00 D-1, normal EXAA 10:15 publication assumed by 11:00 but per-record timestamps missing',
'formula':'price = EXAA_DE + coefficient * (DK1_previous_day - DE_previous_day)','candidates':[0,.1,.25,.5,.75,1.0],
'interval':'paired seven-day block bootstrap; exploratory and not adjusted for model searches'},indent=2)+'\n',encoding='utf8')
print('selected',selected);print(tab.to_string(index=False));print(combined.to_string(index=False))
