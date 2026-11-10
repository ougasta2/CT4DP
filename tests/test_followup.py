import importlib.util
from pathlib import Path

import pandas as pd


def test_matched_cells_compares_within_same_calendar_stratum():
    spec = importlib.util.spec_from_file_location('followup', Path(__file__).parents[1] / 'scripts/run_followup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({'PriceArea':['DK1']*4, 'season':['summer']*3+['winter'],
                          'time_of_day':['afternoon']*4, 'cluster':[2,2,0,0],
                          'negative_price':[True,False,False,True]})
    rows = module.matched_cells(frame, 2, 2024)
    summer = next(row for row in rows if row['season'] == 'summer')
    winter = next(row for row in rows if row['season'] == 'winter')
    assert (summer['solar_n'], summer['solar_events'], summer['other_n']) == (2, 1, 1)
    assert summer['rate_difference'] == .5
    assert pd.isna(winter['rate_difference'])
