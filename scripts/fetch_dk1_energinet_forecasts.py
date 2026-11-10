"""Fetch Energinet DK1/DK2 day-ahead renewable forecast historical snapshots.
The DayAhead field is retrospective, and its pre-auction vintage is UNVERIFIED.
"""
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests
import time

ROOT = Path('data/external/energinet_forecasts')
URL = 'https://api.energidataservice.dk/dataset/Forecasts_Hour'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for year in range(2020, 2025):
        path = ROOT / f'forecasts_{year}.json.gz'
        if path.exists():
            print('cached',path,flush=True)
            continue
        params={'start':f'{year}-01-01','end':f'{year+1}-01-01',
                'filter':json.dumps({'PriceArea':['DK1','DK2']}), 'limit':0}
        for attempt in range(8):
            response=requests.get(URL,params=params,timeout=180)
            if response.status_code != 429:
                break
            delay=int(response.headers.get('Retry-After','30'))+2
            print('rate limited, retry in seconds',delay,flush=True)
            time.sleep(delay)
        response.raise_for_status()
        data=response.json()
        if not data.get('records'):
            raise ValueError('Empty Energinet response')
        payload={'url':response.url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                 'sha256':hashlib.sha256(response.content).hexdigest(),'body':data}
        with gzip.open(path,'wt',encoding='utf8') as f:json.dump(payload,f,separators=(',',':'))
        print(year, len(data['records']),path,flush=True)

if __name__=='__main__':main()
