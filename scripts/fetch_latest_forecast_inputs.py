"""Acquire 2026 sources for latest-complete-day forecast backtesting."""
from pathlib import Path
from fetch_forecast_inputs import get,ROOT,SITES,VARS
from fetch_multiarea import retrieve,validate_energy

END='2026-11-09'  # Last completed Copenhagen calendar day on 2026-11-10.


def main():
    for zone in ['DK1','DK2','DE-LU','NL','SE4']:
        retrieve('https://api.energy-charts.info/v2/price',
                 {'bzn':zone,'start':'2026-01-01','end':END},
                 Path('data/multiarea/energycharts')/f'price_{zone}_2026_to_1109.json.gz',validate_energy)
    for name,(latitude,longitude) in SITES.items():
        p=dict(latitude=latitude,longitude=longitude,start_date='2025-12-30',end_date=END,
               hourly=','.join(VARS),models='jma_gsm',timezone='GMT')
        get('https://previous-runs-api.open-meteo.com/v1/forecast',p,ROOT/'weather'/f'{name}_jma_day2_2026_to_1109.json.gz')
    get('https://fred.stlouisfed.org/graph/fredgraph.csv',
        {'id':'DCOILBRENTEU','cosd':'2018-01-01','coed':END},ROOT/'fred'/'brent_to_20261109.csv.gz')


if __name__=='__main__':main()
