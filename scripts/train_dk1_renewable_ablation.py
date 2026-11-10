"""Hindsight-only Energinet renewable forecast information-value diagnostic.
Official metadata says ForecastDayAhead is published at 18:00 D-1, after our
10:00 cutoff. Models including it are not valid pre-auction predictors.
"""
import gzip,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_weather_ablation as w
import train_dk1_rolling_benchmark as roll

OUT=Path('results/dk1_renewable_ablation')
ROOT=Path('data/external/energinet_forecasts')
TYPES=['Onshore Wind','Offshore Wind','Solar']


def load_renewables(index):
    parts=[]
    for year in range(2020,2025):
        with gzip.open(ROOT/f'forecasts_{year}.json.gz','rt',encoding='utf8') as f:
            obj=json.load(f)
        records=pd.DataFrame(obj['body']['records'])
        records['HourUTC']=pd.to_datetime(records.HourUTC,utc=True)
        records=records[records.PriceArea.isin(['DK1','DK2']) & records.ForecastType.isin(TYPES)]
        records['feature']=records.PriceArea+'_'+records.ForecastType.str.replace(' ','_')
        if records.duplicated(['HourUTC','feature']).any():raise ValueError('Duplicate forecast target/area/type')
        parts.append(records.pivot(index='HourUTC',columns='feature',values='ForecastDayAhead'))
    df=pd.concat(parts).sort_index()
    if df.index.has_duplicates:raise ValueError('Overlapping forecast years')
    df=df.reindex(index)
    expected=[z+'_'+t.replace(' ','_') for z in ['DK1','DK2'] for t in TYPES]
    return df.reindex(columns=expected).apply(pd.to_numeric,errors='raise')


def curves(hourly):
    local=hourly.index.tz_convert('Europe/Copenhagen')
    day=local.tz_localize(None).normalize()
    hours=local.hour
    data=hourly.copy();data['day']=day;data['hour']=hours
    days=pd.DatetimeIndex(sorted(day.unique()))
    fields=[]
    count=data.groupby('day').size()
    for col in hourly.columns:
        complete=data[col].notna().groupby(data.day).all().reindex(days,fill_value=False)
        curve=data.pivot_table(index='day',columns='hour',values=col,aggfunc='mean').reindex(index=days,columns=range(24))
        for date in curve.index[(count.reindex(days)==23)&complete]:
            if curve.loc[date].isna().sum()==1 and pd.isna(curve.loc[date,2]):
                curve.loc[date,2]=(curve.loc[date,1]+curve.loc[date,3])/2
        curve.loc[~complete]=np.nan
        curve.columns=[f'tso_{col}_h{h}' for h in range(24)]
        fields.append(curve)
    return pd.concat(fields,axis=1)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    prices=base.load_prices();orig,y,data=base.make_daily(prices)
    weather,_=w.weather_curves(prices);fuel=roll.commodity_features(orig.index)
    extended=roll.price_features(prices,orig)
    tso=curves(load_renewables(prices.index))
    x0=pd.concat([extended,weather,fuel],axis=1)
    x1=pd.concat([x0,tso],axis=1)
    count=data.groupby('day').size().reindex(orig.index)
    expected=pd.Series([(pd.Timestamp(d+pd.Timedelta(days=1),tz='Europe/Copenhagen').tz_convert('UTC')-pd.Timestamp(d,tz='Europe/Copenhagen').tz_convert('UTC')).total_seconds()/3600 for d in orig.index],index=orig.index)
    common=x1.notna().all(axis=1)&y.notna().all(axis=1)&count.eq(expected)
    days=orig.index[common]
    pd.DataFrame({'day':orig.index,'common_complete':common.to_numpy()}).to_csv(OUT/'daily_coverage.csv',index=False)
    allp=[];snapshots={}
    for month in pd.date_range('2023-01-01','2025-01-01',freq='MS',inclusive='left'):
        end=month+pd.offsets.MonthBegin(1)
        ev=days[(days>=month)&(days<end)]
        if len(ev)==0:raise ValueError('No common evaluation days for '+str(month))
        tr=days[days<month]
        if len(tr)<365:raise ValueError('Insufficient common training days')
        actual=data[data.day.isin(ev)][['day','hour','DK1']].copy()
        lookup=pd.DatetimeIndex(actual.day);h=actual.hour.to_numpy()
        actual['previous_day']=orig.loc[lookup].to_numpy()[np.arange(len(actual)),h]
        for family,kind,window in [('ridge','ridge',None),('lear_raw_3y','lasso',1095)]:
            window_days=tr if window is None else tr[tr>=month-pd.Timedelta(days=window)]
            for label,x in [('without_tso',x0),('with_post_cutoff_tso',x1)]:
                model=roll.fit_model(x.loc[window_days],y.loc[window_days],kind,False)
                curve=roll.predict_model(model,x.loc[ev])
                actual[family+'_'+label]=pd.DataFrame(curve,index=ev).loc[lookup].to_numpy()[np.arange(len(actual)),h]
                if month==pd.Timestamp('2024-12-01'):
                    snapshots[family+'_'+label]={'estimator':model,'feature_columns':list(x.columns),'trained_through':str(tr[-1].date())}
        allp.append(actual)
        print('completed',month.date(),'training days',len(tr),'eval days',len(ev),flush=True)
    pred=pd.concat(allp).sort_index();pred.to_csv(OUT/'predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[];ci=[]
    for year,g in pred.groupby(pred.day.dt.year):
        for name in ['previous_day','ridge_without_tso','ridge_with_post_cutoff_tso','lear_raw_3y_without_tso','lear_raw_3y_with_post_cutoff_tso']:
            rows.append({'year':year,'model':name,**base.metric(g.DK1,g[name])})
        for family in ['ridge','lear_raw_3y']:
            # moving block requires all calendar days to be present
            dates=pd.date_range(g.day.min(),g.day.max(),freq='D')
            if g.day.nunique()==len(dates):
                bounds=w.paired_intervals(g,family+'_without_tso',family+'_with_post_cutoff_tso')
                ci.append({'year':year,'model':family,'mae_improvement_low':bounds[0,0], 'mae_improvement_high':bounds[1,0],
                           'hit_rate_improvement_low':bounds[0,1],'hit_rate_improvement_high':bounds[1,1]})
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(ci).to_csv(OUT/'paired_intervals.csv',index=False)
    joblib.dump(snapshots,OUT/'models.joblib',compress=3)
    protocol={'test':'2023 and 2024 calendar years, evaluated on common complete days',
              'inputs':'same extended price/weather and lagged gas/Brent on common training days, +/- DK1/DK2 wind/solar forecasts',
              'tso':'Energinet Forecasts_Hour ForecastDayAhead; official metadata says published 18:00 D-1, after the 10:00 cutoff',
              'limitation':'Hindsight-only diagnostic: with_post_cutoff_tso models leak post-cutoff information and are not valid pre-auction predictors. No load forecast; no pre-auction capacity or EUA input.',
              'retraining':'monthly; ridge all common days, Lasso trailing 1095 calendar days; same samples for each paired comparison',
              'delivery':'hourly means; DST repeated hour averaged and spring hour interpolated',
              'price_weather_fuel':'JMA 48h archive; lagged gas proxy and Brent; publication and revisions unverified'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(ci).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
