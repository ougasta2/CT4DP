"""Cache 2026 Danish hourly settlement data (revisions not archived)."""
import gzip,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import requests

OUT=Path('data/external/energinet_settlement_2026.json.gz')
if not OUT.exists():
    u='https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement'
    fields=['HourUTC','PriceArea','GrossConsumptionMWh','OffshoreWindLt100MW_MWh','OffshoreWindGe100MW_MWh','OnshoreWindLt50kW_MWh','OnshoreWindGe50kW_MWh','SolarPowerLt10kW_MWh','SolarPowerGe10Lt40kW_MWh','SolarPowerGe40kW_MWh','SolarPowerSelfConMWh']
    r=requests.get(u,params={'start':'2026-01-01','end':'2026-11-02','limit':0,'columns':','.join(fields),
                             'filter':json.dumps({'PriceArea':['DK1','DK2']})},timeout=120)
    r.raise_for_status();d=r.json()
    if not d.get('records'):raise ValueError('Empty 2026 data')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(OUT,'wt',encoding='utf8') as f:json.dump({'url':r.url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
      'sha256':hashlib.sha256(r.content).hexdigest(),'body':d},f,separators=(',',':'))
    print('saved',len(d['records']),OUT)
