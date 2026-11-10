"""Create auditable hourly paired data from cached multi-area source responses."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path('data/multiarea')
OUT=Path('results/multiarea')
YEARS=range(2019,2026)


def payload(path):
    with gzip.open(path,'rt',encoding='utf-8') as f: return json.load(f)['data']


def hourly(frame, fields, mode, cadence):
    """Accept exactly one hourly record or four aligned quarter records per UTC hour."""
    frame=frame.copy()
    frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True)
    if frame.timestamp.duplicated().any(): raise ValueError('Duplicate UTC timestamps')
    if not frame.timestamp.dt.second.eq(0).all(): raise ValueError('Nonaligned seconds')
    frame['hour']=frame.timestamp.dt.floor('h')
    frame['minute']=frame.timestamp.dt.minute
    if not frame.minute.isin([0,15,30,45]).all(): raise ValueError('Nonaligned minutes')
    groups=frame.groupby('hour',sort=True)
    count=groups.minute.size()
    minutes=groups.minute.agg(lambda x: tuple(sorted(x)))
    if cadence == 'price':
        quarter = minutes.index >= pd.Timestamp('2025-09-30T22:00:00Z')
    elif cadence == 'esett':
        quarter = minutes.index >= pd.Timestamp('2023-05-21T22:00:00Z')
    elif cadence in (15,60):
        quarter = np.full(len(minutes), cadence == 15)
    else:
        raise ValueError(f'Unexpected source cadence {cadence}')
    valid=(minutes.map(lambda x: x == (0,15,30,45)) & quarter) | (minutes.map(lambda x: x == (0,)) & ~quarter)
    result=pd.DataFrame(index=count.index)
    result['source_records']=count
    result['valid_resolution']=valid
    for field in fields:
        if field not in frame: frame[field]=np.nan
        frame[field]=pd.to_numeric(frame[field],errors='coerce')
        group=frame.groupby('hour')[field]
        values=group.sum(min_count=1) if mode=='energy' else group.mean()
        result[field]=values.where(valid & group.count().eq(count))
    return result


def chart(kind, label, fields):
    parts=[]
    for year in YEARS:
        obj=payload(ROOT/'energycharts'/f'{kind}_{label}_{year}.json.gz')
        items=[dict(timestamp=x['timestamp'],**x['values']) for x in obj['data']]
        f=pd.DataFrame(items)
        cadence='price' if kind=='price' else obj.get('interval_minutes')
        h=hourly(f,fields,'mean',cadence)
        parts.append(h)
    result=pd.concat(parts).sort_index()
    if result.index.duplicated().any(): raise ValueError('Duplicate yearly boundary')
    return result


def esett(kind, fields):
    parts=[]
    filenames=[ROOT/'esett'/f'SE4_{kind}_boundary_2018.json.gz']
    filenames += [ROOT/'esett'/f'SE4_{kind}_{year}.json.gz' for year in YEARS]
    for path in filenames:
        obj=payload(path)
        f=pd.DataFrame(obj).rename(columns={'timestamp':'local_timestamp','timestampUTC':'timestamp'})
        if not f.mba.eq('SE4').all(): raise ValueError('Unexpected aggregated area')
        parts.append(hourly(f,fields,'energy','esett'))
    result=pd.concat(parts).sort_index()
    if result.index.duplicated().any(): raise ValueError('Duplicate eSett annual boundary')
    return result


def local_year(index): return index.tz_convert('Europe/Copenhagen').year


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for path in sorted(ROOT.rglob('*.json.gz')):
        with gzip.open(path,'rt',encoding='utf-8') as f:
            envelope=json.load(f)
        manifest.append(dict(path=path.as_posix(),url=envelope['url'],
            retrieved_utc=envelope['retrieved_utc'],response_sha256=envelope['sha256'],
            gzip_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope='local_only' if 'esett' in path.parts else 'cc_by_4.0'))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    series={}
    for label in ['DE-LU','NL','SE4']:
        series['price_'+label]=chart('price',label,['day_ahead_price'])
    for label in ['de','nl']:
        series['power_'+label]=chart('public_power',label,['wind_onshore','wind_offshore','solar','load'])
    series['esett_production']=esett('production',['wind','solar','total'])
    series['esett_consumption']=esett('consumption',['total'])
    coverage=[]
    for name,df in series.items():
        for year in YEARS:
            start=pd.Timestamp(f'{year}-01-01',tz='Europe/Copenhagen').tz_convert('UTC')
            end=pd.Timestamp(f'{year+1}-01-01',tz='Europe/Copenhagen').tz_convert('UTC')
            expected=pd.date_range(start,end,freq='h',inclusive='left')
            x=df.reindex(expected)
            cols=[c for c in x.columns if c not in ['source_records','valid_resolution']]
            coverage.append(dict(source=name,year=year,expected=len(x),timestamps_present=int(x.source_records.notna().sum()),
                                 complete=int(x[cols].notna().all(axis=1).sum()),
                                 missing_by_field=json.dumps(x[cols].isna().sum().to_dict(),sort_keys=True),
                                 invalid_resolution=int((x.valid_resolution==False).sum())))
    pd.DataFrame(coverage).to_csv(OUT/'coverage_by_year.csv',index=False)
    old=pd.read_csv('results/decade/processed.csv.gz',parse_dates=['HourUTC'])
    old=old[old.year.isin(YEARS)].set_index('HourUTC')
    rows=[]
    for area,other,power in [('DK1','DE-LU','de'),('DK1','NL','nl'),('DK2','SE4','SE4')]:
        dk=old[old.PriceArea.eq(area)][['SpotPriceEUR','wind_mwh','solar_harmonized_total_mwh','gross_load_mwh']]
        dk=dk.rename(columns={'SpotPriceEUR':'price_dk','wind_mwh':'wind_dk_mwh',
                              'solar_harmonized_total_mwh':'solar_dk_mwh','gross_load_mwh':'load_dk_mwh'})
        price=series['price_'+other][['day_ahead_price']].rename(columns={'day_ahead_price':'price_neighbor'})
        if power=='SE4':
            production=series['esett_production'][['wind','solar','total']].rename(columns={'wind':'wind_neighbor','solar':'solar_neighbor','total':'generation_neighbor'})
            demand=series['esett_consumption'][['total']].rename(columns={'total':'load_neighbor'})
            demand['load_neighbor']=-demand['load_neighbor']
            features=production.join(demand,how='outer')
        else:
            power_df=series['power_'+power]
            features=pd.DataFrame(index=power_df.index)
            features['wind_neighbor']=power_df[['wind_onshore','wind_offshore']].sum(axis=1,min_count=2)
            features['solar_neighbor']=power_df.solar
            features['load_neighbor']=power_df.load
        x=dk.join(price,how='left').join(features,how='left')
        x.index.name='HourUTC';x['year']=local_year(x.index);x['pair']=area+'--'+other
        x['valid_price']=x[['price_dk','price_neighbor']].notna().all(axis=1)
        required=['price_dk','price_neighbor','wind_dk_mwh','solar_dk_mwh','load_dk_mwh','wind_neighbor','solar_neighbor','load_neighbor']
        x['usable_features']=(np.isfinite(x[required]).all(axis=1)&x.load_dk_mwh.gt(0)&x.load_neighbor.gt(0)
                              &x[['wind_dk_mwh','solar_dk_mwh','wind_neighbor','solar_neighbor']].ge(0).all(axis=1))
        x.to_csv(ROOT/f'paired_{area}_{other}_2019_2025.csv.gz',compression={'method':'gzip','mtime':0})
        for year,g in x.groupby('year'):
            z=g[g.usable_features]; neg_a=z.price_dk.lt(0);neg_b=z.price_neighbor.lt(0)
            either=neg_a|neg_b; discord=neg_a^neg_b
            days=z.index.tz_convert('Europe/Copenhagen').date
            rows.append(dict(pair=area+'--'+other,year=int(year),hours=len(g),valid_prices=int(g.valid_price.sum()),
                usable=int(g.usable_features.sum()),any_negative=int(either.sum()),discordant=int(discord.sum()),
                negative_days=len(set(days[either])),discordant_days=len(set(days[discord]))))
    pd.DataFrame(rows).to_csv(OUT/'pair_coverage_events.csv',index=False)


if __name__=='__main__': main()
