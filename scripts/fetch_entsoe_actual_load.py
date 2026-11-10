"""Download ENTSO-E A65/A04? actual-load backfills for Danish zones.
Source is ENTSO-E 6.1.A, not the day-ahead forecast or settlement.
"""
import gzip,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import requests
ROOT=Path('data/external/entsoe_actual_load_archive')
BASE='https://raw.githubusercontent.com/Energy-Data-Science/entsoe-realtime-data/data/data/raw'
ROOT.mkdir(parents=True,exist_ok=True)
for zone in ['DK1','DK2']:
 for year in range(2021,2027):
  target=ROOT/f'{zone}_{year}.json.gz'
  if target.exists():continue
  url=f'{BASE}/{zone}/actual_load/{year}.csv';r=requests.get(url,timeout=45);r.raise_for_status()
  with gzip.open(target,'wt',encoding='utf8') as f:json.dump({'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
      'sha256':hashlib.sha256(r.content).hexdigest(),'body':r.text},f,separators=(',',':'))
  print(zone,year,len(r.content),flush=True)
