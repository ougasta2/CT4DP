"""Small, cached official API client. API date bounds are Copenhagen local time."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = 'https://api.energidataservice.dk'
PRODUCTION = 'ProductionConsumptionSettlement'
PRICES = 'Elspotprices'
KEYS = ['HourUTC', 'PriceArea']
WIND = ['OffshoreWindLt100MW_MWh', 'OffshoreWindGe100MW_MWh',
        'OnshoreWindLt50kW_MWh', 'OnshoreWindGe50kW_MWh']
SOLAR = ['SolarPowerLt10kW_MWh', 'SolarPowerGe10Lt40kW_MWh', 'SolarPowerGe40kW_MWh']
SELF = 'SolarPowerSelfConMWh'
COLUMNS = {PRODUCTION: KEYS + WIND + SOLAR + [SELF, 'GrossConsumptionMWh'],
           PRICES: KEYS + ['SpotPriceEUR']}


class Client:
    def __init__(self, cache='data/cache', offline=False):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.session = requests.Session()

    def get(self, path, params=None):
        params = params or {}
        key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()[:24]
        target = self.cache / (key + '.json')
        if target.exists():
            return json.loads(target.read_text(encoding='utf-8'))['payload']
        if self.offline:
            raise FileNotFoundError(f'Offline cache miss: {BASE}/{path}; params={params}')
        for attempt in range(4):
            try:
                response = self.session.get(f'{BASE}/{path}', params=params, timeout=(15, 90))
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 3:
                        response.raise_for_status()
                    wait = response.headers.get('Retry-After', '')
                    delay = float(wait) if wait.isdigit() else 2 ** (attempt + 1)
                    if delay > 600:
                        raise RuntimeError(f'Rate limit requires {delay}s; rerun later: {response.url}')
                    print(f'{path}: rate limited; respecting Retry-After={delay}s', flush=True)
                    while delay > 0:
                        chunk = min(delay, 30)
                        time.sleep(chunk)
                        delay -= chunk
                    continue
                response.raise_for_status()
                payload = response.json()
                envelope = dict(url=response.url, parameters=params,
                                downloaded_utc=datetime.now(timezone.utc).isoformat(), payload=payload)
                temp = target.with_suffix('.tmp')
                temp.write_text(json.dumps(envelope), encoding='utf-8')
                temp.replace(target)
                time.sleep(0.3)
                return payload
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f'API request exhausted retries: {path}')

    def metadata(self, dataset):
        meta = self.get(f'meta/dataset/{dataset}')
        fields = [c['dbColumn'] for c in meta['columns']]
        missing = set(COLUMNS[dataset]) - set(fields)
        if missing:
            raise ValueError(f'{dataset}: unexpected schema, missing {missing}')
        print(f'{dataset} metadata schema: {fields}', flush=True)
        return meta

    def window(self, dataset, start, end):
        params = dict(start=start, end=end, filter=json.dumps({'PriceArea': ['DK1', 'DK2']}),
                      columns=','.join(COLUMNS[dataset]), sort='HourUTC ASC,PriceArea ASC', limit=50000)
        payload = self.get(f'dataset/{dataset}', params)
        records = payload.get('records')
        if not records:
            raise ValueError(f'{dataset}: no records for {start} to {end}')
        if len(records) >= 50000 or payload.get('total', len(records)) != len(records):
            raise ValueError('Window truncated; use shorter intervals rather than partial data')
        frame = pd.DataFrame(records)
        if set(COLUMNS[dataset]) - set(frame):
            raise ValueError(f'{dataset}: response schema mismatch')
        print(f'{dataset} {start} .. {end}: {len(frame)} rows', flush=True)
        return frame

    def year(self, dataset, year):
        # One bounded historical year is <18k area-hours. Fewer requests respect
        # the particularly strict discontinued-price dataset rate limit.
        return self.window(dataset, f'{year}-01-01', f'{year+1}-01-01')
