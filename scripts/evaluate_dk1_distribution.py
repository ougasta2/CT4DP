"""Compare rolling empirical predictive distributions based on two point models."""
from pathlib import Path
import pandas as pd
import train_dk1_preauction as base
import train_dk1_rolling_benchmark as roll

OUT=Path('results/dk1_rolling_benchmark')
p=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC')
rows=[]
for name in ['lear_ensemble','lear_raw_3y','ridge_original_weather']:
    data=p[['day','hour','DK1',name]].copy().rename(columns={name:'lear_ensemble'})
    calibrated=roll.apply_distribution(data)
    for year,g in calibrated[calibrated.day.dt.year.isin([2023,2024])].groupby(calibrated.day.dt.year):
        z=g[g.interval_point.notna()]
        rows.append({'year':year,'distribution_base':name,**base.metric(z.DK1,z.interval_point),
                     'coverage_90':((z.DK1>=z.q05)&(z.DK1<=z.q95)).mean(),
                     'width_90':(z.q95-z.q05).mean(),'crps':z.crps.mean()})
    calibrated.to_csv(OUT/f'distribution_{name}.csv.gz',compression={'method':'gzip','mtime':0})
result=pd.DataFrame(rows);result.to_csv(OUT/'distribution_comparison.csv',index=False)
print(result.to_string(index=False))
