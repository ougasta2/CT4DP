"""Inventory shared original inputs and derived ten-year artifacts with checksums."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def entry(path):
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    cached=[]
    for path in sorted((ROOT/'data/cache').glob('*.json')):
        envelope=json.loads(path.read_text(encoding='utf-8'))
        payload=envelope['payload']
        row=entry(path)
        row.update(source_url=envelope['url'],parameters=envelope['parameters'],
                   downloaded_utc=envelope['downloaded_utc'],rows=len(payload['records']) if 'records' in payload else None)
        cached.append(row)
    outputs=[entry(path) for path in sorted((ROOT/'data/decade').glob('*.csv.gz'))]
    for filename in ['processed.csv.gz','holdout_predictions.csv.gz','fitted_models.joblib','frozen_protocol.json']:
        path=ROOT/'results/decade'/filename
        if path.exists():outputs.append(entry(path))
    manifest=dict(scope='2016-2025 completed Copenhagen calendar years, DK1 and DK2; all original selected-field study inputs',
                  train_years=list(range(2016,2024)),test_years=[2024,2025],
                  data_source='Energinet Energi Data Service',source_data_license='CC BY 4.0',
                  attribution='Source: Energinet (www.energidataservice.dk). Derived tables and models are transformations; no endorsement implied.',
                  source_data_license_url='https://creativecommons.org/licenses/by/4.0/',
                  cached_responses=cached,derived_artifacts=outputs)
    (ROOT/'data/decade/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'Indexed {len(cached)} cached responses and {len(outputs)} derived artifacts.')


if __name__=='__main__':main()
