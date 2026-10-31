"""Export the complete study inputs and inventory from cached official responses."""
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from energy_pilot.api import Client,PRODUCTION,PRICES
from energy_pilot.data import keyed


def main():
    client=Client(ROOT/'data/cache',offline=True)
    exports=[]
    for dataset,filename in [(PRODUCTION,'production_consumption_2023_2024.csv.gz'),
                             (PRICES,'elspotprices_2023_2024.csv.gz')]:
        frame=keyed(pd.concat([client.year(dataset,year) for year in [2023,2024]],ignore_index=True))
        frame=frame.sort_values(['HourUTC','PriceArea']).reset_index(drop=True)
        assert len(frame)==35088, f'{dataset}: unexpected study coverage'
        path=ROOT/'data'/filename
        frame.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
        exports.append(dict(path=path.relative_to(ROOT).as_posix(),dataset=dataset,rows=len(frame),
                            columns=list(frame),time_column='HourUTC (UTC)',
                            local_start_inclusive='2023-01-01 Europe/Copenhagen',
                            local_end_exclusive='2025-01-01 Europe/Copenhagen'))
    files=[]
    for path in sorted((ROOT/'data/cache').glob('*.json')):
        envelope=json.loads(path.read_text(encoding='utf-8'))
        parsed=urlparse(envelope['url'])
        assert parsed.scheme=='https' and parsed.netloc=='api.energidataservice.dk'
        assert not parsed.username and not parsed.password
        payload=envelope['payload']
        files.append(dict(path=path.relative_to(ROOT).as_posix(),source_url=envelope['url'],
                          parameters=envelope['parameters'],downloaded_utc=envelope['downloaded_utc'],
                          rows=len(payload['records']) if 'records' in payload else None,
                          columns=list(payload['records'][0]) if payload.get('records') else
                                  [c['dbColumn'] for c in payload.get('columns',[])]))
    for item in files+exports:
        path=ROOT/item['path']
        item.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest=dict(source='Energinet Energi Data Service',license='CC BY 4.0',
                  license_url='https://creativecommons.org/licenses/by/4.0/',
                  attribution='Source: Energinet (www.energidataservice.dk). CSV exports and analysis are derived transformations; no endorsement implied.',
                  scope='Complete downloaded inputs used by this DK1/DK2 2023–2024 study; selected source columns only.',
                  exports=exports,cached_responses=files)
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Exported 35,088 rows per dataset; inventoried {len(files)} original cached responses.')


if __name__=='__main__':
    main()
