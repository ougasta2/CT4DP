import importlib.util
from pathlib import Path
import pandas as pd

spec=importlib.util.spec_from_file_location('prepare_multiarea', Path(__file__).parents[1]/'scripts/prepare_multiarea.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def test_price_quarters_are_complete_and_averaged():
    f=pd.DataFrame({'timestamp':['2025-09-30T21:00Z','2025-09-30T22:00Z','2025-09-30T22:15Z',
                                 '2025-09-30T22:30Z','2025-09-30T22:45Z','2025-09-30T23:00Z'],
                    'price':[10.,-4.,0.,4.,8.,999.]})
    x=mod.hourly(f,['price'],'mean','price')
    assert x.price.iloc[0]==10. and x.price.iloc[1]==2.
    assert pd.isna(x.price.iloc[2]) and not x.valid_resolution.iloc[2]


def test_energy_quarters_sum_and_missing_values_are_not_zeroed():
    f=pd.DataFrame({'timestamp':pd.date_range('2024-01-01',periods=8,freq='15min',tz='UTC'),
                    'wind':[1.,2.,3.,4.,1.,None,3.,4.]})
    x=mod.hourly(f,['wind'],'energy','esett')
    assert x.wind.iloc[0]==10. and pd.isna(x.wind.iloc[1])
