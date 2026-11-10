"""Build DK1/DK2 demand forecasts without target-day demand leakage.

The statistical specification follows the temperature/calendar/lag regression
family in Hong et al., Global Energy Forecasting Competition 2012 (2014),
https://doi.org/10.1016/j.ijforecast.2013.07.001, with a 24-output ridge fit.
Settlement observations are lagged >=14 days; publication/revision histories
are not verified, so these are conservative proxies rather than audited vintages.
"""
import gzip,io,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_latest_forecast as latest

OUT=Path('results/dk1_demand_forecast')
ROOT=Path('data/external/entsoe_load_archive')
ZONES=['DK1','DK2']


def load_realized():
    x=pd.read_csv('results/decade/processed.csv.gz',parse_dates=['HourUTC'])
    x=x[x.PriceArea.isin(ZONES)].copy()
    source=Path('data/external/energinet_settlement_2026.json.gz')
    if source.exists():
        with gzip.open(source,'rt',encoding='utf8') as f:records=json.load(f)['body']['records']
        recent=pd.DataFrame(records).rename(columns={'GrossConsumptionMWh':'gross_load_mwh'})
        recent['HourUTC']=pd.to_datetime(recent.HourUTC,utc=True)
        x=pd.concat([x[['HourUTC','PriceArea','gross_load_mwh']],recent[['HourUTC','PriceArea','gross_load_mwh']]],ignore_index=True)
    if x.duplicated(['PriceArea','HourUTC']).any():raise ValueError('Duplicate settlement timestamp')
    z=x.pivot(index='HourUTC',columns='PriceArea',values='gross_load_mwh').sort_index()
    z=z.where(z>0)
    return z


def load_archived_tso(zone):
    frames=[]
    for year in range(2021,2026):
        with gzip.open(ROOT/f'{zone}_{year}.json.gz','rt',encoding='utf8') as f:obj=json.load(f)
        df=pd.read_csv(io.StringIO(obj['body']),parse_dates=['timestamp_utc'])
        if not df.country.eq(zone).all() or not df.variable.eq('forecast_load').all():raise ValueError('Wrong source or zone')
        if df.timestamp_utc.duplicated().any():raise ValueError('Duplicate forecast hour')
        frames.append(df.set_index('timestamp_utc').value)
    return pd.concat(frames).sort_index()


def load_design(load,weather,zone,lags=(14,21,28)):
    local=load.index.tz_convert('Europe/Copenhagen')
    days=local.tz_localize(None).normalize()
    df=pd.DataFrame({'day':days,'hour':local.hour,'load':load.to_numpy()},index=load.index)
    dates=pd.DatetimeIndex(sorted(days.unique()))
    mat=df.pivot_table(index='day',columns='hour',values='load',aggfunc='mean').reindex(index=dates,columns=range(24))
    complete=df.groupby('day').load.apply(lambda s:s.notna().all())
    counts=df.groupby('day').size()
    expected=pd.Series([(pd.Timestamp(d+pd.Timedelta(days=1),tz='Europe/Copenhagen').tz_convert('UTC')-pd.Timestamp(d,tz='Europe/Copenhagen').tz_convert('UTC')).total_seconds()/3600 for d in dates],index=dates)
    complete=complete.reindex(dates,fill_value=False)&counts.reindex(dates).eq(expected)
    for day in mat.index[(counts.reindex(dates)==23)&complete]:
        if pd.isna(mat.loc[day,2]) and mat.loc[day].isna().sum()==1:
            mat.loc[day,2]=(mat.loc[day,1]+mat.loc[day,3])/2
    mat.loc[~complete]=np.nan
    inputs=[]
    for lag in lags:
        g=mat.shift(lag).copy();g.columns=[f'load_lag{lag}_h{i}' for i in range(24)];inputs.append(g)
    w=weather.reindex(load.index).copy();w['day']=days;w['hour']=local.hour
    for variable in ['temperature_2m','wind_speed_10m','cloud_cover']:
        good=w.groupby('day')[variable].apply(lambda s:s.notna().all()).reindex(dates,fill_value=False)
        v=w.pivot_table(index='day',columns='hour',values=variable,aggfunc='mean').reindex(index=dates,columns=range(24))
        for day in v.index[(counts.reindex(dates)==23)&good]:
            if pd.isna(v.loc[day,2]) and v.loc[day].isna().sum()==1:v.loc[day,2]=(v.loc[day,1]+v.loc[day,3])/2
        v.loc[~good]=np.nan
        v.columns=[f'{variable}_h{i}' for i in range(24)];inputs.append(v)
    design=pd.concat(inputs,axis=1)
    design['weekday_sin']=np.sin(2*np.pi*dates.dayofweek/7);design['weekday_cos']=np.cos(2*np.pi*dates.dayofweek/7)
    design['year_sin']=np.sin(2*np.pi*dates.dayofyear/365.25);design['year_cos']=np.cos(2*np.pi*dates.dayofyear/365.25)
    return design,mat,df


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    load=load_realized();rows=[];predictions=[];availability=[]
    for zone in ZONES:
        x,y,data=load_design(load[zone],latest.weather_all(zone),zone)
        valid=x.notna().all(axis=1)&y.notna().all(axis=1)
        dates=x.index[valid]
        tso=load_archived_tso(zone).rename('tso_backfill')
        for month in pd.date_range('2021-01-01','2026-11-01',freq='MS',inclusive='left'):
            end=month+pd.offsets.MonthBegin(1)
            test=dates[(dates>=month)&(dates<end)]
            # Dated settlement labels >=10 days old at first issue of the month.
            train=dates[dates<month-pd.Timedelta(days=10)]
            if len(train)<365 or len(test)==0:continue
            model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
            model.fit(x.loc[train],y.loc[train])
            curve=pd.DataFrame(model.predict(x.loc[test]),index=test,columns=range(24))
            p=data[data.day.isin(test)].copy();key=pd.DatetimeIndex(p.day);h=p.hour.to_numpy()
            p['forecast']=curve.loc[key].to_numpy()[np.arange(len(p)),h]
            p['naive14']=x.loc[key].to_numpy()[np.arange(len(p)),h]
            p['zone']=zone;p['train_cutoff']=str((month-pd.Timedelta(days=10)).date())
            p=p.join(tso,how='left');predictions.append(p)
            availability.append({'zone':zone,'month':str(month.date()),'train_days':len(train),'last_train_target':str(train[-1].date()),'test_days':len(test)})
        print('completed demand forecasts',zone,flush=True)
    result=pd.concat(predictions).sort_index()
    if result.duplicated(['zone','day','hour']).any():
        # Duplicate local hour is legitimate autumn DST; distinct UTC required.
        if result.reset_index().duplicated(['zone','HourUTC']).any():raise ValueError('Duplicate UTC')
    result.rename_axis('HourUTC').to_csv(OUT/'demand_forecasts.csv.gz',compression={'method':'gzip','mtime':0})
    pd.DataFrame(availability).to_csv(OUT/'fit_manifest.csv',index=False)
    for (year,zone),g in result.groupby([result.day.dt.year,'zone']):
        for name in ['naive14','forecast','tso_backfill']:
            h=g[g[name].notna()&g.load.notna()]
            if len(h):rows.append({'year':year,'zone':zone,'model':name,'hours':len(h),'mae_mw':float(np.abs(h.load-h[name]).mean()),'rmse_mw':float(np.sqrt(np.mean((h.load-h[name])**2)))})
    pd.DataFrame(rows).to_csv(OUT/'demand_metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    spec={'target':'DK1/DK2 hourly gross consumption from settlement, MWh/h equivalent MW',
      'model':'24-output Ridge over three 14/21/28-day lagged consumption curves, 48h nominal weather forecasts and calendar; refit monthly',
      'issue':'D-1 10:00 Europe/Copenhagen', 'training_label_cutoff':'before first day of forecast month minus 10 days',
      'settlement':'Energinet first publishes DataHub settlement after 8 days; 14-day lag is conservative, actual historical revisions/release timestamps unverified',
      'weather':'JMA previous_day2 nominal 48h lead; publication/revision not individually audited',
      'official_backfill':'ENTSO-E A65 retrieved via third-party 2026 historical backfill; current/revised values, not individual pre-auction vintages',
      'price_use':'only generated out-of-fold demand forecasts are candidates for price model; never use contemporaneous realized consumption',
      'citation':'https://doi.org/10.1016/j.ijforecast.2013.07.001; model is an adaptation, not exact paper reproduction'}
    (OUT/'protocol.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
