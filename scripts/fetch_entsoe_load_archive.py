"""Download public ENTSO-E load forecast backfills; historical vintages unverified."""
import gzip,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import requests

ROOT=Path('data/external/entsoe_load_archive')
BASE='https://raw.githubusercontent.com/Energy-Data-Science/entsoe-realtime-data/data/data/raw'

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for zone in ['DK1','DK2']:
        for year in range(2021,2026):
            path=ROOT/f'{zone}_{year}.json.gz'
            if path.exists():continue
            url=f'{BASE}/{zone}/forecast_load/{year}.csv'
            r=requests.get(url,timeout=45);r.raise_for_status()
            envelope={'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
              'sha256':hashlib.sha256(r.content).hexdigest(),'body':r.text}
            with gzip.open(path,'wt',encoding='utf8') as f:json.dump(envelope,f,separators=(',',':'))
            print(zone,year,len(r.content),flush=True)

if __name__=='__main__':main()
