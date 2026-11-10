"""Short-lag demand forecasts using ENTSO-E 6.1.A actual load.

DK1/DK2 ENTSO-E documentation says values are published at latest H+1 and not
subsequently updated. Individual historical publication instants are not in
this backfill, so punctual availability is an assumption, not an audit.
"""
import gzip,io,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import train_dk1_demand_forecast as d
import train_dk1_preauction as base
import train_latest_forecast as latest

ROOT=Path('data/external/entsoe_actual_load_archive')
OUT=Path('results/dk1_fast_demand')


def historical(zone):
    parts=[]
    for year in range(2021,2027):
        with gzip.open(ROOT/f'{zone}_{year}.json.gz','rt',encoding='utf8') as f:j=json.load(f)
        a=pd.read_csv(io.StringIO(j['body']),parse_dates=['timestamp_utc'])
        if not a.country.eq(zone).all() or not a.variable.eq('actual_load').all():raise ValueError('wrong data source')
        if a.timestamp_utc.duplicated().any():raise ValueError('duplicate actual load')
        parts.append(a.set_index('timestamp_utc').value)
    s=pd.concat(parts).sort_index()
    if s.index.has_duplicates:raise ValueError('duplicate years')
    return pd.to_numeric(s,errors='raise').where(lambda x:x>0)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    prices=base.load_prices();pieces=[];fits=[]
    for zone in ['DK1','DK2']:
        load=historical(zone).reindex(prices.index)
        x,y,data=d.load_design(load,latest.weather_all(zone),zone,lags=(2,7,14))
        # Predictions need only known inputs; training additionally requires observed labels.
        can_predict=x.notna().all(axis=1);can_train=can_predict&y.notna().all(axis=1)
        for month in pd.date_range('2022-02-01','2026-09-01',freq='MS',inclusive='left'):
            end=month+pd.offsets.MonthBegin(1)
            ev=x.index[can_predict&(x.index>=month)&(x.index<end)]
            # Entire training delivery day must have ended at least one day
            # before D-1 issue day, leaving >H+1 before the 10:00 issue.
            tr=x.index[can_train&(x.index<month-pd.Timedelta(days=1))]
            if len(ev)==0 or len(tr)<300:continue
            model=make_pipeline(StandardScaler(),Ridge(alpha=100.))
            model.fit(x.loc[tr],y.loc[tr]);curve=pd.DataFrame(model.predict(x.loc[ev]),index=ev,columns=range(24))
            out=data[data.day.isin(ev)].copy();key=pd.DatetimeIndex(out.day);hours=out.hour.to_numpy()
            out['forecast']=curve.loc[key].to_numpy()[np.arange(len(out)),hours];out['zone']=zone
            pieces.append(out);fits.append({'zone':zone,'month':str(month.date()),'train_days':len(tr),'last_train_day':str(tr[-1].date()),'forecast_days':len(ev)})
        print('completed fast demand',zone,flush=True)
    out=pd.concat(pieces).sort_index()
    out.rename_axis('HourUTC').to_csv(OUT/'forecasts.csv.gz',compression={'method':'gzip','mtime':0})
    pd.DataFrame(fits).to_csv(OUT/'fit_manifest.csv',index=False)
    rows=[]
    for (year,zone),g in out.groupby([out.day.dt.year,'zone']):
        good=g.load.notna();rows.append({'year':year,'zone':zone,'hours':int(good.sum()),'mae_mw':float((g.loc[good,'load']-g.loc[good,'forecast']).abs().mean())})
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    (OUT/'protocol.json').write_text(json.dumps({'target':'ENTSO-E 6.1.A DK1/DK2 actual load',
      'model':'24-output ridge using past load day-2, day-7, day-14 plus nominal 48h weather and calendar',
      'cutoff':'D-1 10:00 Copenhagen; last prior full day D-2 ends by D-1 midnight and is due at H+1',
      'source':'https://transparencyplatform.zendesk.com/hc/en-us/articles/16647979768084-Actual-Total-Load-Day-ahead-Per-Bidding-Zone-6-1-A-6-1-B',
      'limitations':['H+1 and no later updates are documented at source but individual historic publication timestamps absent','weather forecast issue/revision timestamps unverified','Third-party archive ends in August 2026']},indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    with threadpool_limits(limits=2):run()
