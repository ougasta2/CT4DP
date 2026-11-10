"""Audit a bounded 2024 sample of Energy-Charts source coverage and licensing."""
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = 'https://api.energy-charts.info/v2/'
CASES = [('price', {'bzn': z}) for z in ['DK1','DK2','DE-LU','NL','SE3','SE4','NO2']]
CASES += [('public_power', {'country': c}) for c in ['dk','de','nl','se','no']]


def run():
    out = Path('results/neighbor_audit'); out.mkdir(parents=True, exist_ok=True)
    raw = Path('data/neighbor_sample'); raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); summaries = []; fields = []; manifest = []
    for endpoint, params in CASES:
        query = {**params, 'start':'2024-01-01', 'end':'2024-12-31'}
        label = next(iter(params.values()))
        print(endpoint, label, flush=True)
        response = None
        try:
            for attempt in range(5):
                response = session.get(BASE+endpoint, params=query, timeout=90)
                if response.status_code not in (429, 500, 502, 503, 504): break
                time.sleep(2**attempt * 2)
            response.raise_for_status()
            j = response.json()
            frame = pd.DataFrame([{'timestamp':x['timestamp'], **x['values']} for x in j['data']])
            ts = pd.to_datetime(frame.timestamp, utc=True)
            interval = int(j['interval_minutes'])
            tz = j['timezone']
            start = pd.Timestamp('2024-01-01',tz=tz).tz_convert('UTC')
            end = pd.Timestamp('2025-01-01',tz=tz).tz_convert('UTC')
            expected = pd.date_range(start,end,freq=f'{interval}min',inclusive='left')
            inside = ts[(ts >= start) & (ts < end)]
            license_info = j.get('license','')
            summaries.append(dict(endpoint=endpoint,label=label,area_scope=j.get('bidding_zone'),
                country=j.get('country'),resolution=j.get('resolution'),unit=j.get('unit'),
                rows=len(frame), expected=len(expected), unique_in_year=inside.nunique(),
                missing_timestamps=len(expected.difference(inside)),duplicates=int(inside.duplicated().sum()),
                license=license_info, status='ok'))
            for col in frame.columns.drop('timestamp') if 'CC BY 4.0' in license_info else []:
                values = pd.to_numeric(frame[col],errors='coerce')
                fields.append(dict(endpoint=endpoint,label=label,field=col,n=len(values),
                                   nonnull=int(values.notna().sum()),null=int(values.isna().sum()),
                                   mean=float(values.mean()) if values.notna().any() else None,
                                   sum=float(values.sum()) if values.notna().any() else None))
            # Store only explicitly CC BY 4.0 responses; restricted payloads are not redistributed.
            if 'CC BY 4.0' in license_info:
                path = raw/f'{endpoint}_{label}_2024.json'
                path.write_bytes(response.content)
                sha = hashlib.sha256(response.content).hexdigest()
                manifest.append(dict(endpoint=endpoint,label=label,url=response.url,license=license_info,
                                     path=str(path),sha256=sha))
        except (requests.RequestException,ValueError,KeyError,TypeError) as exc:
            summaries.append(dict(endpoint=endpoint,label=label,status='error',error=str(exc)))
        time.sleep(1)
    pd.DataFrame(summaries).to_csv(out/'coverage_2024.csv',index=False)
    pd.DataFrame(fields).to_csv(out/'field_coverage_2024.csv',index=False)
    (out/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__': run()
