"""Cache public day-ahead NTC backfills, not point-in-time vintages."""
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path('data/external/entsoe_ntc_backfill')
URL = 'https://entsoemcp.com/api/series'
ROOT.mkdir(parents=True, exist_ok=True)
session = requests.Session()
for origin, destination in [('DK_1', 'DE_LU'), ('DE_LU', 'DK_1')]:
    for year in [2026, 2022, 2023, 2024, 2025]:
      for month in range(1, 13):
        path = ROOT / f'{origin}_{destination}_{year}-{month:02}.json.gz'
        if path.exists():
            continue
        params = {'endpoint': 'net_transfer_capacity_dayahead', 'from_zone': origin,
                  'to_zone': destination, 'start': f'{year}-{month:02}-01',
                  'end': (f'{year+1}-01-01' if month == 12 else f'{year}-{month+1:02}-01'),
                  'aggregation': 'raw'}
        for attempt in range(6):
            response = session.get(URL, params=params, timeout=90)
            if response.status_code not in [429, 503]:
                response.raise_for_status()
                break
            delay = max(60, min(int(response.headers.get('Retry-After', '60')), 300))
            print('capacity retry', response.status_code, 'after', delay, 'seconds', flush=True)
            time.sleep(delay)
        else:
            raise RuntimeError(f'Capacity service still unavailable: {response.status_code}')
        body = response.json()
        if body.get('endpoint') != params['endpoint'] or 'data' not in body or body.get('truncated'):
            raise ValueError('Unexpected ENTSO-E capacity response')
        payload = {'url': response.url, 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                   'sha256': hashlib.sha256(response.content).hexdigest(), 'body': body}
        with gzip.open(path, 'wt', encoding='utf8') as handle:
            json.dump(payload, handle, separators=(',', ':'))
        print(origin, destination, year, month, len(body['data']), flush=True)
        time.sleep(2)
