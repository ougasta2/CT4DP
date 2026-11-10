"""Rolling pre-auction proxy forecasts of wind/solar generation (not TSO data)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import train_latest_forecast as latest
import train_price_regression as old

OUT=Path('results/dk1_supply_forecast')


def feature_frame(zone):
    raw=pd.read_csv('results/decade/processed.csv.gz',parse_dates=['HourUTC'])
    raw=raw[raw.PriceArea.eq(zone)].sort_values('HourUTC').reset_index(drop=True)
    if not raw.HourUTC.diff().dropna().eq(pd.Timedelta(hours=1)).all():raise ValueError('missing settlement hour')
    local=raw.HourUTC.dt.tz_convert('Europe/Copenhagen')
    x=pd.DataFrame(index=raw.index)
    for source in ['wind_mwh','solar_harmonized_total_mwh']:
        temp=raw[['HourUTC',source]].rename(columns={source:'price'})
        for lag in [14,21,28]:x[source+'_lag'+str(lag)]=old.local_lag(temp,lag)
    weather=latest.weather_all(zone).reindex(pd.DatetimeIndex(raw.HourUTC))
    for name in ['temperature_2m','wind_speed_10m','cloud_cover']:
        x[name]=weather[name].to_numpy()
    x['hour_sin']=np.sin(2*np.pi*local.dt.hour/24);x['hour_cos']=np.cos(2*np.pi*local.dt.hour/24)
    x['year_sin']=np.sin(2*np.pi*local.dt.dayofyear/365.25);x['year_cos']=np.cos(2*np.pi*local.dt.dayofyear/365.25)
    raw['day']=pd.to_datetime(local.dt.date)
    return raw,x


def run():
    OUT.mkdir(parents=True,exist_ok=True);outputs=[];rows=[]
    for zone in ['DK1','DK2']:
        raw,x=feature_frame(zone)
        good=np.isfinite(x.to_numpy()).all(axis=1)
        zoneparts=[]
        for month in pd.date_range('2021-01-01','2026-01-01',freq='MS',inclusive='left'):
            cutoff=month-pd.Timedelta(days=10)
            train=good&(raw.day<cutoff)&(raw.day>=cutoff-pd.Timedelta(days=730))
            ev=good&(raw.day>=month)&(raw.day<month+pd.offsets.MonthBegin(1))
            if train.sum()<5000 or ev.sum()==0:continue
            z=raw.loc[ev,['HourUTC','day','wind_mwh','solar_harmonized_total_mwh']].copy();z['zone']=zone
            for field in ['wind_mwh','solar_harmonized_total_mwh']:
                ok=train&raw[field].notna()&raw[field].ge(0)
                model=HistGradientBoostingRegressor(loss='squared_error',max_iter=80,learning_rate=.06,
                   max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=5,early_stopping=False,random_state=42)
                model.fit(x.loc[ok],raw.loc[ok,field]);z['forecast_'+field]=np.maximum(model.predict(x.loc[ev]),0)
            zoneparts.append(z)
        part=pd.concat(zoneparts);outputs.append(part)
        for year,g in part.groupby(part.day.dt.year):
            for field in ['wind_mwh','solar_harmonized_total_mwh']:
                ok=g[field].notna();rows.append({'zone':zone,'year':year,'variable':field,'hours':int(ok.sum()),
                    'mae_mw':float((g.loc[ok,field]-g.loc[ok,'forecast_'+field]).abs().mean())})
        print('supply forecast complete',zone,flush=True)
    result=pd.concat(outputs).sort_values(['HourUTC','zone'])
    result.to_csv(OUT/'forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    (OUT/'protocol.json').write_text(json.dumps({'model':'HGB on 14/21/28-day settlement generation, calendar and 48h nominal weather forecast',
      'cutoff':'D-1 10:00 Copenhagen','training':'2y rolling, refit monthly; last training target older than first day of month minus 10 days',
      'caveats':['not official TSO forecast','publication and revision histories of weather and settlement not audited','no target-day realized generation used as model input'],
      'output':'hourly mean wind and solar generation forecasts, two zones'},indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
