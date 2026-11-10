"""Describe 2024 joint negative-price counts using redistributable samples."""
import json
from pathlib import Path
import pandas as pd


def load_price(zone):
    path = Path('data/neighbor_sample')/f'price_{zone}_2024.json'
    obj = json.loads(path.read_text(encoding='utf-8'))
    if 'CC BY 4.0' not in obj.get('license',''): raise ValueError('Unexpected source license')
    rows = [{'timestamp':v['timestamp'],'price':v['values']['day_ahead_price']} for v in obj['data']]
    frame = pd.DataFrame(rows)
    frame.timestamp = pd.to_datetime(frame.timestamp,utc=True)
    if frame.timestamp.duplicated().any() or frame.price.isna().any(): raise ValueError('Invalid price sample')
    return frame.set_index('timestamp').price


def run():
    zones = {z:load_price(z) for z in ['DK1','DK2','DE-LU','NL','SE4','NO2']}
    rows = []
    for a,b in [('DK1','DE-LU'),('DK2','DE-LU'),('DK1','NL'),('DK2','SE4'),('DK1','NO2')]:
        pair = pd.concat([zones[a].rename('a'),zones[b].rename('b')],axis=1,join='inner').dropna()
        x,y = pair.a.lt(0),pair.b.lt(0)
        rows.append(dict(area_a=a,area_b=b,n=len(pair),a_only=int((x&~y).sum()),
                         b_only=int((~x&y).sum()),both=int((x&y).sum()),
                         any_negative=int((x|y).sum()),discordant=int((x^y).sum())))
    out=Path('results/neighbor_audit'); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'paired_events_2024.csv',index=False)
    old = pd.read_csv('results/decade/processed.csv.gz',usecols=['HourUTC','year','PriceArea','SpotPriceEUR'])
    old = old[old.year.eq(2024)]; old['HourUTC'] = pd.to_datetime(old.HourUTC,utc=True)
    checks=[]
    for z in ['DK1','DK2']:
        original=old[old.PriceArea.eq(z)].set_index('HourUTC').SpotPriceEUR
        both=pd.concat([original.rename('old'),zones[z].rename('new')],axis=1,join='inner').dropna()
        checks.append(dict(area=z,n=len(both),negative_disagreements=int((both.old.lt(0)!=both.new.lt(0)).sum()),
                           max_absolute_eur_mwh=float((both.old-both.new).abs().max())))
    pd.DataFrame(checks).to_csv(out/'dk_price_reconciliation.csv',index=False)


if __name__=='__main__': run()
