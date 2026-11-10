"""Acquire 2019-2025 public inputs, preserving raw responses and source metadata."""
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path('data/multiarea')
SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'DTU-student-energy-study/1.0 (research; reproducible audit)'


def retrieve(url, params, target, validate, pause=1):
    if target.exists():
        print('cached', target, flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(5):
        try:
            r = SESSION.get(url, params=params, timeout=120)
            if r.status_code in (429,500,502,503,504):
                delay = max(2**attempt*3, int(r.headers.get('Retry-After','0')) if r.headers.get('Retry-After','').isdigit() else 0)
                print('retry', r.status_code, 'seconds', delay, flush=True)
                time.sleep(min(delay,60)); continue
            r.raise_for_status()
            parsed = r.json()
            validate(parsed)
            payload = dict(url=r.url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                           sha256=hashlib.sha256(r.content).hexdigest(), data=parsed)
            part = target.with_suffix(target.suffix+'.part')
            with gzip.open(part,'wt',encoding='utf-8') as f: json.dump(payload,f,separators=(',',':'))
            part.replace(target)
            print('saved', target, flush=True)
            time.sleep(pause)
            return
        except (requests.RequestException,ValueError,KeyError,TypeError) as e:
            error = str(e)
            print('retry', error[:160], flush=True)
            time.sleep(2**attempt*3)
    raise RuntimeError(f'Failed {url} {params}: {error or "retry limit"}')


def validate_energy(j):
    if not isinstance(j,dict) or not j.get('data') or not j.get('series'):
        raise ValueError('Empty or invalid Energy-Charts response')
    if 'CC BY 4.0' not in j.get('license',''):
        raise ValueError('Unexpected/non-redistributable source license')


def validate_esett(j):
    if not isinstance(j,list) or not j or not all('timestampUTC' in x for x in j):
        raise ValueError('Empty or invalid eSett response')


def main():
    # Local 2019-01-01 starts at 2018-12-31 23:00 UTC.
    for name,endpoint in [('production','EXP16/Volumes'),('consumption','EXP15/Consumption')]:
        params={'mba':'10Y1001A1001A47J',
                'start':'2018-12-31T23:00:00.000Z','end':'2019-01-01T00:00:00.000Z'}
        retrieve('https://api.opendata.esett.com/'+endpoint,params,
                 ROOT/'esett'/f'SE4_{name}_boundary_2018.json.gz',validate_esett)
    for year in range(2019,2026):
        for endpoint, key, values in [('price','bzn',['DE-LU','NL','SE4']),
                                      ('public_power','country',['de','nl'])]:
            for value in values:
                params={key:value,'start':f'{year}-01-01','end':f'{year}-12-31'}
                path=ROOT/'energycharts'/f'{endpoint}_{value}_{year}.json.gz'
                retrieve('https://api.energy-charts.info/v2/'+endpoint,params,path,validate_energy)
        for name,endpoint in [('production','EXP16/Volumes'),('consumption','EXP15/Consumption')]:
            params={'mba':'10Y1001A1001A47J','start':f'{year}-01-01T00:00:00.000Z',
                    'end':f'{year+1}-01-01T00:00:00.000Z'}
            path=ROOT/'esett'/f'SE4_{name}_{year}.json.gz'
            retrieve('https://api.opendata.esett.com/'+endpoint,params,path,validate_esett)


if __name__=='__main__': main()
