"""Vanilla load model, implementing Ziel & Liu (2016), equations 10–11.
https://doi.org/10.1016/j.ijforecast.2016.01.001
Categories are one-hot, and a small ridge penalty is added for stability.
The paper's probabilistic scenario generation is not implemented here.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_dk1_demand_forecast as base
import train_latest_forecast as latest

OUT=Path('results/dk1_demand_forecast')


def design(index,temperature):
    local=index.tz_convert('Europe/Copenhagen')
    m=np.eye(12)[local.month-1];w=np.eye(7)[local.weekday];h=np.eye(24)[local.hour]
    t=np.asarray(temperature,dtype=float)/10.0
    t3=np.column_stack([t,t*t,t*t*t])
    # Eq. 10: month, weekday, hour, weekday×hour; Eq. 11: T, T², T³
    # with month and hour interactions. One-hot expansion handles factors.
    arr=np.column_stack([np.ones(len(index)),m,w,h,(w[:,:,None]*h[:,None,:]).reshape(len(index),-1),
           t3,(t3[:,:,None]*m[:,None,:]).reshape(len(index),-1),
           (t3[:,:,None]*h[:,None,:]).reshape(len(index),-1)])
    return arr


def run():
    load=base.load_realized();results=[];fits=[]
    for zone in base.ZONES:
        index=load.index;local=index.tz_convert('Europe/Copenhagen');day=local.tz_localize(None).normalize()
        temp=latest.weather_all(zone).reindex(index).temperature_2m.to_numpy()
        x=design(index,temp);y=load[zone].to_numpy();input_valid=np.isfinite(x).all(axis=1)
        finite=np.isfinite(y)&input_valid
        eye=np.eye(x.shape[1]);eye[0,0]=0
        for month in pd.date_range('2021-01-01','2026-01-01',freq='MS',inclusive='left'):
            train_end=month-pd.Timedelta(days=10)
            train_start=train_end-pd.DateOffset(years=2)
            fit=finite&(day>=train_start)&(day<train_end)
            ev=input_valid&(day>=month)&(day<month+pd.offsets.MonthBegin(1))
            if fit.sum()<8000 or ev.sum()==0:continue
            xt=x[fit];yt=y[fit]
            coef=np.linalg.solve(xt.T@xt+eye,xt.T@yt)
            z=pd.DataFrame({'HourUTC':index[ev],'day':day[ev],'hour':local.hour[ev],
                'zone':zone,'load':y[ev],'vanilla_forecast':x[ev]@coef})
            results.append(z);fits.append({'zone':zone,'month':str(month.date()),'fit_hours':int(fit.sum()),'train_cutoff':str(train_end.date())})
        print('finished vanilla demand',zone,flush=True)
    result=pd.concat(results,ignore_index=True)
    result.to_csv(OUT/'vanilla_forecasts.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(fits).to_csv(OUT/'vanilla_fit_manifest.csv',index=False)
    rows=[]
    for (year,zone),g in result.groupby([result.day.dt.year,'zone']):
        rows.append({'year':year,'zone':zone,'hours':len(g),'mae_mw':float(np.abs(g.load-g.vanilla_forecast).mean()),
                     'rmse_mw':float(np.sqrt(np.mean((g.load-g.vanilla_forecast)**2)))})
    pd.DataFrame(rows).to_csv(OUT/'vanilla_metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
