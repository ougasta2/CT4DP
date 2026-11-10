"""Apply the frozen 2023–24-selected nonlinear design to Jan–Oct 2026."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_rolling_benchmark as roll
import train_dk1_weather_ablation as weather
import train_dk1_demand_price as experiment
import train_dk1_nonlinear as nonlinear

OUT=Path('results/dk1_demand_price')


def run():
    price=base.load_prices();orig,y,raw=base.make_daily(price)
    w,_=weather.weather_curves(price);d=experiment.forecast_curves(price.index)['forecast']
    fuel=roll.commodity_features(orig.index)
    x=pd.concat([roll.price_features(price,orig),w,d,fuel],axis=1)
    counts=raw.groupby('day').size().reindex(orig.index)
    expected=pd.Series([(pd.Timestamp(day+pd.Timedelta(days=1),tz='Europe/Copenhagen').tz_convert('UTC')-pd.Timestamp(day,tz='Europe/Copenhagen').tz_convert('UTC')).total_seconds()/3600 for day in orig.index],index=orig.index)
    complete=x.notna().all(axis=1)&y.notna().all(axis=1)&counts.eq(expected)
    days=orig.index[complete];outputs=[];snapshots={}
    for month in pd.date_range('2026-01-01','2026-11-01',freq='MS',inclusive='left'):
        ev=days[(days>=month)&(days<month+pd.offsets.MonthBegin(1))];tr=days[days<month]
        if len(ev)!=(month+pd.offsets.MonthBegin(1)-month).days:raise ValueError('missing '+str(month))
        fitted=roll.fit_model(x.loc[tr],y.loc[tr],'lasso',False)
        curve=pd.DataFrame(roll.predict_model(fitted,x.loc[ev]),index=ev)
        z=raw[raw.day.isin(ev)][['day','hour','DK1']].copy();key=pd.DatetimeIndex(z.day);h=z.hour.to_numpy()
        z['lasso_own_demand_fuel']=curve.loc[key].to_numpy()[np.arange(len(z)),h]
        z['previous_day']=orig.loc[key].to_numpy()[np.arange(len(z)),h]
        outputs.append(z)
        if month==pd.Timestamp('2026-10-01'):snapshots['lasso']={'estimator':fitted,'features':list(x.columns),'train_end':str(tr[-1].date())}
    fresh=pd.concat(outputs).sort_index()
    past=pd.read_csv(OUT/'predictions.csv.gz',parse_dates=['HourUTC','day']).set_index('HourUTC').sort_index()
    past=past[['day','hour','DK1','previous_day','lasso_own_demand_fuel']]
    allp=pd.concat([past,fresh]).sort_index();feat=nonlinear.design(allp)
    results=[]
    for month in pd.date_range('2026-01-01','2026-11-01',freq='MS',inclusive='left'):
        train=allp.day.lt(month)&allp.day.ge(month-pd.Timedelta(days=1095));ev=allp.day.ge(month)&allp.day.lt(month+pd.offsets.MonthBegin(1))
        z=allp.loc[ev].copy()
        model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,learning_rate=.04,max_leaf_nodes=15,
                min_samples_leaf=100,l2_regularization=5.,early_stopping=False,random_state=42)
        model.fit(feat.loc[train],allp.DK1.loc[train]);z['hgb_price']=model.predict(feat.loc[ev]);results.append(z)
        if month==pd.Timestamp('2026-10-01'):snapshots['hgb']={'estimator':model,'features':list(feat.columns),'train_end':str(allp.day.loc[train].max().date())}
        print('2026 evaluation',month.date(),flush=True)
    result=pd.concat(results).sort_index();result.to_csv(OUT/'evaluation_2026_predictions.csv.gz',compression={'method':'gzip','mtime':0})
    metrics=pd.DataFrame([{'year':2026,'model':name,**base.metric(result.DK1,result[name])} for name in ['previous_day','lasso_own_demand_fuel','hgb_price']])
    metrics.to_csv(OUT/'evaluation_2026_metrics.csv',index=False)
    joblib.dump(snapshots,OUT/'evaluation_2026_models.joblib',compress=3)
    (OUT/'evaluation_2026_protocol.json').write_text(json.dumps({'evaluation':'2026-01-01 to 2026-10-31, 304 local days',
      'selected':'hgb_price selected using 2023 and 2024; design improvements evaluated on 2025 before this run',
      'selection_warning':'2026 calendar period was examined in earlier project experiments and is not pristine',
      'demand':'realized settlement is only modelled using 10-day training cutoff and 14+ day input lags',
      'caveats':['Weather, settlement, fuel and price revisions/publication times individually unverified.','Hourly means, not 15-minute prices.']},indent=2)+'\n',encoding='utf8')
    print(metrics.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
