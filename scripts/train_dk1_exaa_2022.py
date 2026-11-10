"""11:00 pre-auction 2022 EXAA information experiment, separate from 10:00.
EXAA 10:15 quarter-hour auction in APG official archive, 2021–2022 only.
A timely 11:00 publication is assumed, not proven per historical record.
"""
import io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_demand_price as d
import train_dk1_weather_ablation as weather
import train_dk1_rolling_benchmark as roll

OUT=Path('results/dk1_exaa_2022')


def early_prices():
    outer=zipfile.ZipFile('data/external/apg_day_ahead_history.zip');parts=[]
    for year in [2021,2022]:
        name=[n for n in outer.namelist() if f'_{year}_English.zip' in n]
        if len(name)!=1:raise ValueError('unexpected APG archive layout')
        nested=zipfile.ZipFile(io.BytesIO(outer.read(name[0])))
        f=pd.read_csv(io.BytesIO(nested.read(nested.namelist()[0])))
        raw=f['Time from [CET/CEST]'].str.replace(' 2A:',' 02:',regex=False).str.replace(' 2B:',' 02:',regex=False)
        utc=pd.to_datetime(raw,errors='raise').dt.tz_localize('Europe/Vienna',ambiguous='infer',nonexistent='raise').dt.tz_convert('UTC')
        z=pd.DataFrame({'utc':utc,'price':pd.to_numeric(f['Price EXAA 10:15 Auction [EUR/MWh]'],errors='raise')})
        if z.utc.duplicated().any():raise ValueError('duplicate EXAA quarter')
        z['hour']=z.utc.dt.floor('h');z['minute']=z.utc.dt.minute
        minutes=z.groupby('hour').minute.agg(lambda s:tuple(sorted(s)))
        if not minutes.map(lambda m:m==(0,15,30,45)).all():raise ValueError('incomplete EXAA quarter')
        parts.append(z.groupby('hour').price.mean())
    series=pd.concat(parts).sort_index()
    if series.index.has_duplicates:raise ValueError('EXAA overlapping years')
    return series


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    prices=base.load_prices();original,y,data=base.make_daily(prices)
    w,_=weather.weather_curves(prices);demand=d.forecast_curves(prices.index)['forecast'];fuel=roll.commodity_features(original.index)
    x=pd.concat([roll.price_features(prices,original),w,demand,fuel],axis=1)
    source=early_prices().reindex(prices.index)
    local=prices.index.tz_convert('Europe/Copenhagen');idx=local.tz_localize(None).normalize()
    temp=pd.DataFrame({'day':idx,'hour':local.hour,'exaa':source.to_numpy()},index=prices.index)
    dates=original.index;curve=temp.pivot_table(index='day',columns='hour',values='exaa',aggfunc='mean').reindex(index=dates,columns=range(24))
    good=temp.groupby('day').exaa.apply(lambda s:s.notna().all()).reindex(dates,fill_value=False)
    count=temp.groupby('day').size().reindex(dates)
    for day in curve.index[(count==23)&good]:
        if curve.loc[day].isna().sum()==1 and pd.isna(curve.loc[day,2]):curve.loc[day,2]=(curve.loc[day,1]+curve.loc[day,3])/2
    curve.loc[~good]=np.nan
    curve.columns=[f'exaa_at_early_h{i}' for i in range(24)]
    xe=pd.concat([x,curve],axis=1)
    good=xe.notna().all(axis=1)&y.notna().all(axis=1)&(count>=23)
    days=dates[good];parts=[]
    for month in pd.date_range('2022-01-01','2023-01-01',freq='MS',inclusive='left'):
        tr=days[days<month];ev=days[(days>=month)&(days<month+pd.offsets.MonthBegin(1))]
        if len(tr)<300 or len(ev)==0:raise ValueError('Insufficient matching historical days')
        z=data[data.day.isin(ev)][['day','hour','DK1']].copy();ix=pd.DatetimeIndex(z.day);h=z.hour.to_numpy()
        z['previous_day']=original.loc[ix].to_numpy()[np.arange(len(z)),h]
        z['exaa_raw']=source.reindex(z.index).to_numpy()
        for family in ['ridge','lasso']:
            for feature,xx in [('without_exaa',x),('with_exaa',xe)]:
                fitted=roll.fit_model(xx.loc[tr],y.loc[tr],family,False)
                prediction=pd.DataFrame(roll.predict_model(fitted,xx.loc[ev]),index=ev)
                z[f'{family}_{feature}']=prediction.loc[ix].to_numpy()[np.arange(len(z)),h]
        parts.append(z);print('EXAA',month.date(),flush=True)
    result=pd.concat(parts).sort_index();result.to_csv(OUT/'predictions.csv.gz',compression={'method':'gzip','mtime':0})
    rows=[]
    for name in ['previous_day','exaa_raw','ridge_without_exaa','ridge_with_exaa','lasso_without_exaa','lasso_with_exaa']:
        rows.append({'year':2022,'model':name,**base.metric(result.DK1,result[name])})
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps({'cutoff':'11:00 Copenhagen on D-1, not the previous 10:00 task',
      'source':'APG archive, explicitly labelled Price EXAA 10:15 Auction, AT; 15m mean to hour',
      'source_url':'https://markt.apg.at/en/transparency/transmission/day-ahead-prices/',
      'training':'matched observed 2021 days, monthly refit during 2022', 'evaluation':'2022 historical benchmark',
      'availability':'11:00 availability of early auction is assumed based on normal schedule; per-day publication and revisions not audited',
      'weather_and_demand':'historical vintages individually unverified',
      'excluded_candidate':'2023/24/25 Salzburg EXAA-labeled series matched 12:00 market coupling 744/744 hours for January 2024 and is never included',
      'caveat':'2022 one-year test only, market conditions extreme; not proof of 2023+ performance'},indent=2)+'\n',encoding='utf8')
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
