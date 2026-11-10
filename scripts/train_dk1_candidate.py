"""Fit candidate DK1 estimators through the newest complete archived day."""
import json
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_dk1_weather_ablation as w
import train_dk1_rolling_benchmark as roll

OUT=Path('results/dk1_rolling_benchmark')


def run():
    prices=base.load_prices();orig,y,data=base.make_daily(prices)
    weather,_=w.weather_curves(prices);fuel=roll.commodity_features(orig.index)
    x0=pd.concat([orig,weather],axis=1)
    x1=pd.concat([roll.price_features(prices,orig),weather,fuel],axis=1)
    counts=data.groupby('day').size().reindex(orig.index)
    expected=pd.Series([(pd.Timestamp(d+pd.Timedelta(days=1),tz='Europe/Copenhagen').tz_convert('UTC')-pd.Timestamp(d,tz='Europe/Copenhagen').tz_convert('UTC')).total_seconds()/3600 for d in orig.index],index=orig.index)
    valid=x1.notna().all(axis=1)&y.notna().all(axis=1)&counts.eq(expected)
    days=orig.index[valid];last=days[-1];end=last+pd.Timedelta(days=1)
    short=days[days>=end-pd.Timedelta(days=1095)]
    models={}
    for name,x,train,kind in [('ridge',x0,days,'ridge'),('lear_raw_3y',x1,short,'lasso')]:
        fitted=roll.fit_model(x.loc[train],y.loc[train],kind,False)
        models[name]={'estimator':fitted,'feature_columns':list(x.columns),'first_day':str(train[0].date()),'last_day':str(last.date()),'training_days':len(train)}
    joblib.dump(models,OUT/'latest_candidate_models.joblib',compress=3)
    info={'training_last_local_day':str(last.date()),'models':{name:{k:v[k] for k in ['first_day','last_day','training_days']} for name,v in models.items()},
          'note':'Fit artifacts only: no fresh forecast issued. Inputs require new price/weather/fuel data and historic release/vintage verification.',
          'calibration':'No online median-bias adjustment stored; requires recent rolling out-of-sample residual history.'}
    (OUT/'latest_candidate_protocol.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
