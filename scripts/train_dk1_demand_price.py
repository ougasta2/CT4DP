"""DK1 walk-forward electricity price forecasts with out-of-fold demand forecasts.

LEAR-inspired statistical model: for hour h on day d,
p[d,h] = intercept[h] + sum_j beta[h,j] x[d,j] + error[d,h],
where beta minimizes squared error + alpha * L1(beta).
Reference: Lago et al., Applied Energy 293 (2021), 116983,
https://doi.org/10.1016/j.apenergy.2021.116983. This is an adaptation.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_rolling_benchmark as prior
import train_dk1_weather_ablation as weather_src

OUT=Path('results/dk1_demand_price')


def forecast_curves(index, frame=None):
    if frame is None:
        frame=pd.read_csv('results/dk1_demand_forecast/demand_forecasts.csv.gz',parse_dates=['HourUTC','day'])
        vanilla_path=Path('results/dk1_demand_forecast/vanilla_forecasts.csv.gz')
        if vanilla_path.exists():
            vanilla=pd.read_csv(vanilla_path,parse_dates=['HourUTC'])[['HourUTC','zone','vanilla_forecast']]
            if vanilla.duplicated(['HourUTC','zone']).any():raise ValueError('duplicate vanilla demand forecast')
            frame=frame.merge(vanilla,on=['HourUTC','zone'],how='left',validate='one_to_one')
    else:
        frame=frame.copy()
    if frame.duplicated(['zone','HourUTC']).any():raise ValueError('duplicate UTC load forecast')
    # Select only forecast outputs, never contemporaneous realized load.
    sources=['forecast','tso_backfill']+(['vanilla_forecast'] if 'vanilla_forecast' in frame else [])
    frame=frame[['HourUTC','day','hour','zone']+sources]
    local=index.tz_convert('Europe/Copenhagen'); days=pd.DatetimeIndex(sorted(local.tz_localize(None).normalize().unique()))
    count=pd.Series(local.tz_localize(None).normalize()).value_counts().reindex(days)
    groups={}
    for source in sources:
        fields=[]
        for zone in ['DK1','DK2']:
            a=frame[frame.zone.eq(zone)].set_index('HourUTC').reindex(index)
            a['day']=local.tz_localize(None).normalize();a['hour']=local.hour
            good=a.groupby('day')[source].apply(lambda s:s.notna().all()).reindex(days,fill_value=False)
            p=a.pivot_table(index='day',columns='hour',values=source,aggfunc='mean').reindex(index=days,columns=range(24))
            for day in p.index[(count==23)&good]:
                if p.loc[day].isna().sum()==1 and pd.isna(p.loc[day,2]):p.loc[day,2]=(p.loc[day,1]+p.loc[day,3])/2
            p.loc[~good]=np.nan
            p.columns=[f'demand_{source}_{zone}_h{i}' for i in range(24)]
            fields.append(p)
        groups[source]=pd.concat(fields,axis=1)
    return groups


def fit_forecasts():
    OUT.mkdir(parents=True,exist_ok=True)
    prices=base.load_prices();original,y,data=base.make_daily(prices)
    w,_=weather_src.weather_curves(prices)
    demand=forecast_curves(prices.index)
    extended=prior.price_features(prices,original)
    base_input=pd.concat([extended,w],axis=1)
    own=pd.concat([base_input,demand['forecast']],axis=1)
    official=pd.concat([base_input,demand['tso_backfill']],axis=1)
    fuel=prior.commodity_features(original.index)
    inputs={'base':base_input,'own_demand':own,'own_demand_fuel':pd.concat([own,fuel],axis=1),'tso_revised':official}
    if 'vanilla_forecast' in demand:
        inputs['vanilla_demand']=pd.concat([base_input,demand['vanilla_forecast']],axis=1)
    counts=data.groupby('day').size().reindex(original.index)
    expected=pd.Series([(pd.Timestamp(d+pd.Timedelta(days=1),tz='Europe/Copenhagen').tz_convert('UTC')-pd.Timestamp(d,tz='Europe/Copenhagen').tz_convert('UTC')).total_seconds()/3600 for d in original.index],index=original.index)
    common=inputs['own_demand_fuel'].notna().all(axis=1)&official.notna().all(axis=1)&y.notna().all(axis=1)&counts.eq(expected)
    if 'vanilla_demand' in inputs:common &= inputs['vanilla_demand'].notna().all(axis=1)
    dates=original.index[common];pd.DataFrame({'day':original.index,'common':common}).to_csv(OUT/'coverage.csv',index=False)
    predictions=[];snapshots={}
    for month in pd.date_range('2022-01-01','2026-01-01',freq='MS',inclusive='left'):
        end=month+pd.offsets.MonthBegin(1);ev=dates[(dates>=month)&(dates<end)];train=dates[dates<month]
        if len(train)<300 or len(ev)==0:continue
        target=data[data.day.isin(ev)][['day','hour','DK1']].copy()
        lookup=pd.DatetimeIndex(target.day);h=target.hour.to_numpy()
        target['previous_day']=original.loc[lookup].to_numpy()[np.arange(len(target)),h]
        for feature,x in inputs.items():
            for family in ['ridge','lasso']:
                model=prior.fit_model(x.loc[train],y.loc[train],family,False)
                curve=pd.DataFrame(prior.predict_model(model,x.loc[ev]),index=ev)
                target[f'{family}_{feature}']=curve.loc[lookup].to_numpy()[np.arange(len(target)),h]
                if month==pd.Timestamp('2025-12-01'):
                    snapshots[f'{family}_{feature}']={'estimator':model,'features':list(x.columns),'last_train_day':str(train[-1].date())}
        predictions.append(target)
        print('price',month.date(),'train_days',len(train),'eval_days',len(ev),flush=True)
    p=pd.concat(predictions).sort_index()
    p.to_csv(OUT/'predictions.csv.gz',compression={'method':'gzip','mtime':0})
    models=[c for c in p if c not in ['day','hour','DK1']]
    metrics=[]
    for year,g in p.groupby(p.day.dt.year):
        for name in models:
            metrics.append({'year':year,'model':name,**base.metric(g.DK1,g[name])})
    tab=pd.DataFrame(metrics);tab.to_csv(OUT/'metrics.csv',index=False)
    dev=tab[(tab.year.isin([2023,2024]))&(~tab.model.str.contains('tso_revised'))]
    # Fixed guardrail: no worse than Ridge base mean MAE. No choice based on 2025.
    guard=dev[dev.model.eq('ridge_base')].mae.mean()
    candidates=dev.groupby('model').agg(hit=('rate_5pct','mean'),mae=('mae','mean'))
    candidates=candidates[candidates.mae<=guard].sort_values(['hit','mae'],ascending=[False,True])
    selected=candidates.index[0]
    candidates.to_csv(OUT/'development_ranking.csv')
    joblib.dump(snapshots,OUT/'models.joblib',compress=3)
    spec={'benchmark':'2023 and 2024 used for development; 2025 used as held-out year for this selection but viewed in prior studies',
      'primary_metric':'percentage within 5% among absolute prices >=20 EUR/MWh',
      'model_selection':'highest equally-weighted 2023/2024 hit rate subject to mean MAE <= ridge_base; exclude revised official demand',
      'selected':selected,'selection_guardrail_mae':float(guard),
      'demand':'own rolling-origin forecast from lagged settlement and forecast weather, see results/dk1_demand_forecast/protocol.json',
      'tso_revised':'ENTSO-E historical backfill is not proven pre-cutoff and explicitly excluded from selection',
      'causality':'models refit monthly using earlier target days only, own demand forecasts were generated out of fold',
      'limitations':['Historic releases/revisions of weather, price, settlement and fuel not audited.','2025 has been partially viewed in previous experiments.','2025-10 onwards 15-minute prices averaged hourly.','No guarantee of 50% or operational vintage.']}
    (OUT/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf8')
    print('selected',selected,'guardrail',guard,flush=True)
    print(tab[tab.year.isin([2023,2024,2025])].to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):fit_forecasts()
