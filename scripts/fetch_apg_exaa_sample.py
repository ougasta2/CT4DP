"""Cache a fixed one-in-four-day sample of APG's public daily EXAA chart API.
Sample uses ordinal % 4, so all weekdays rotate. Output is local and retains
both the URL and retrieval timestamp; timing is not historical vintage evidence.
"""
import gzip,hashlib,json,time
from datetime import datetime,timezone,date,timedelta
from pathlib import Path
import requests

ROOT=Path('data/external/apg_exaa_sample');ROOT.mkdir(parents=True,exist_ok=True)
first=date(2023,1,1);last=date(2026,1,1);day=first;done=0
session=requests.Session()
while day<last:
 if day.toordinal()%4!=0:
  day+=timedelta(days=1);continue
 target=ROOT/f'{day.isoformat()}.json.gz'
 if target.exists():
  day+=timedelta(days=1);continue
 nxt=day+timedelta(days=1)
 url=f'https://transparency.apg.at/api/v1/EXAAD1P/Data/English/PT60M/{day.isoformat()}T000000/{nxt.isoformat()}T000000/EXAA_Full'
 for attempt in range(5):
  try:
   r=session.get(url,params={'p_exaaMode':'EXAA_Full'},timeout=45)
   if r.status_code==429 or r.status_code>=500:
    delay=min(int(r.headers.get('Retry-After','5')),60);print('retry',day,r.status_code,delay,flush=True);time.sleep(delay);continue
   r.raise_for_status();j=r.json();rows=j['ResponseData']['ValueRows'];cols=j['ResponseData']['ValueColumns']
   if not 23<=len(rows)<=25 or not any(c['InternalName']=='EXAAD1Price' for c in cols):raise ValueError('Bad response shape')
   envelope={'url':r.url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(r.content).hexdigest(),'body':j}
   with gzip.open(target,'wt',encoding='utf8') as f:json.dump(envelope,f,separators=(',',':'))
   break
  except (requests.RequestException,ValueError,KeyError) as err:
   print('error',day,type(err).__name__,str(err)[:100],flush=True)
   if attempt==4:raise
   time.sleep(5)
 done+=1
 if done%20==0:print('APG sampled days downloaded',done,'last',day,flush=True)
 day+=timedelta(days=1);time.sleep(1.0)
print('APG sampled days complete',done,flush=True)
