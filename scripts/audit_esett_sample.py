"""Check Nordic area-level eSett data; no raw payloads are redistributed."""
from pathlib import Path
import pandas as pd
import requests

AREAS = {'SE3':'10Y1001A1001A46L','SE4':'10Y1001A1001A47J','NO2':'10YNO_2________T'}
PERIODS = [('winter','2024-01-01','2024-01-08'),('summer','2024-07-01','2024-07-08')]
ENDPOINTS = {'production':'EXP16/Volumes','consumption':'EXP15/Consumption'}


def run():
    results = []
    for area, code in AREAS.items():
        for season, start, end in PERIODS:
            for kind, endpoint in ENDPOINTS.items():
                print(area, season, kind, flush=True)
                p = {'mba':code,'start':start+'T00:00:00.000Z','end':end+'T00:00:00.000Z'}
                row = dict(area=area,season=season,kind=kind,start=start,end_exclusive=end)
                try:
                    r = requests.get('https://api.opendata.esett.com/'+endpoint,params=p,timeout=45)
                    r.raise_for_status()
                    data = pd.DataFrame(r.json())
                    stamps = pd.to_datetime(data.timestampUTC,utc=True)
                    grid = pd.date_range(start,end,tz='UTC',freq='15min',inclusive='left')
                    inside = stamps[(stamps >= grid[0]) & (stamps <= grid[-1])]
                    row.update(status='ok',rows=len(data),expected_15min=len(grid),
                               unique_in_window=inside.nunique(),duplicates=int(inside.duplicated().sum()),
                               missing_timestamps=len(grid.difference(inside)),
                               returned_mba=','.join(sorted(data.mba.unique())))
                    for col in (['wind','solar','total'] if kind=='production' else ['total']):
                        row[col+'_nonnull'] = int(data[col].notna().sum()) if col in data else 0
                except (requests.RequestException,ValueError,KeyError,TypeError) as e:
                    row.update(status='error',error=str(e))
                results.append(row)
    out = Path('results/neighbor_audit'); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/'esett_samples_2024.csv',index=False)


if __name__=='__main__': run()
