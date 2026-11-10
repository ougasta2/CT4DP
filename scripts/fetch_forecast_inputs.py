"""Fetch past JMA forecast vintages and FRED Brent daily spot series."""
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT=Path('data/external')
SITES={'DK1':(55.476,8.459),'DK2':(55.676,12.568),
       'DE-LU':(53.551,9.994),'NL':(52.367,4.904),'SE4':(55.605,13.003)}
VARS=['temperature_2m_previous_day2','wind_speed_10m_previous_day2','cloud_cover_previous_day2']


def get(url,params,path):
    if path.exists(): print('cached',path,flush=True); return
    path.parent.mkdir(parents=True,exist_ok=True)
    r=requests.get(url,params=params,timeout=180);r.raise_for_status()
    blob={'url':r.url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
          'sha256':hashlib.sha256(r.content).hexdigest(),'body':r.text}
    tmp=path.with_suffix(path.suffix+'.part')
    with gzip.open(tmp,'wt',encoding='utf-8') as f: json.dump(blob,f,separators=(',',':'))
    tmp.replace(path);print('saved',path,flush=True)


def main():
    for name,(latitude,longitude) in SITES.items():
        p=dict(latitude=latitude,longitude=longitude,start_date='2018-12-30',end_date='2025-12-31',
               hourly=','.join(VARS),models='jma_gsm',timezone='GMT')
        get('https://previous-runs-api.open-meteo.com/v1/forecast',p,ROOT/'weather'/f'{name}_jma_day2.json.gz')
    get('https://fred.stlouisfed.org/graph/fredgraph.csv',
        {'id':'DCOILBRENTEU','cosd':'2018-01-01','coed':'2025-12-31'},ROOT/'fred'/'brent.csv.gz')


if __name__=='__main__':main()
