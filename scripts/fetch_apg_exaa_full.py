"""Extend existing APG daily cache to every local day in 2023–2025.
Paced one request/sec; honors server Retry-After. Local-only source cache.
"""
import gzip,hashlib,json,time
from datetime import datetime,timezone,date,timedelta
from pathlib import Path
import requests
ROOT=Path('data/external/apg_exaa_sample');ROOT.mkdir(parents=True,exist_ok=True)
day=date(2023,1,1);last=date(2026,1,1);n=0;session=requests.Session()
while day<last:
 path=ROOT/f'{day.isoformat()}.json.gz'
 if path.exists():day+=timedelta(days=1);continue
 nxt=day+timedelta(days=1)
 u=f'https://transparency.apg.at/api/v1/EXAAD1P/Data/English/PT60M/{day.isoformat()}T000000/{nxt.isoformat()}T000000/EXAA_Full'
 for attempt in range(5):
  try:
   r=session.get(u,params={'p_exaaMode':'EXAA_Full'},timeout=45)
   if r.status_code==429 or r.status_code>=500:
    delay=min(int(r.headers.get('Retry-After','10')),60);print('retry',day,r.status_code,delay,flush=True);time.sleep(delay);continue
   r.raise_for_status();body=r.json()['ResponseData'];rows=body['ValueRows']
   if len(rows) not in [23,24,25] or not any(c['InternalName']=='EXAAD1Price' for c in body['ValueColumns']):raise ValueError('bad response')
   with gzip.open(path,'wt',encoding='utf8') as f:json.dump({'url':r.url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
    'sha256':hashlib.sha256(r.content).hexdigest(),'body':r.json()},f,separators=(',',':'))
   break
  except (requests.RequestException,ValueError,KeyError) as e:
   print('error',day,type(e).__name__,str(e)[:100],flush=True)
   if attempt==4:raise
   time.sleep(5)
 n+=1
 if n%25==0:print('additional APG days',n,'last',day,flush=True)
 day+=timedelta(days=1);time.sleep(1)
print('APG full cache complete, additional days',n,flush=True)
