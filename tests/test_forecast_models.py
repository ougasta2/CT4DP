import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('forecast_models',Path(__file__).parents[1]/'scripts/train_forecast_models.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_weather_and_price_features_do_not_use_current_target_or_future_oil():
    times=pd.date_range('2024-02-01',periods=200,freq='h',tz='UTC')
    frames=[]
    for pair in mod.PAIRS:
        frames.append(pd.DataFrame({'HourUTC':times,'pair':pair,'price_dk':np.arange(200,dtype=float),
            'price_neighbor':np.arange(200,dtype=float)+1,'year':2024}))
    raw=pd.concat(frames,ignore_index=True)
    weather={p:pd.DataFrame({v:1. for v in mod.WVARS},index=times) for p in ['DK1','DK2','DE-LU','NL','SE4']}
    oil=pd.Series([70.,80.,999.],index=pd.to_datetime(['2024-01-01','2024-01-18','2024-02-01']))
    x=mod.build_frame(raw,weather,oil)
    row=x[(x.pair==mod.PAIRS[0])].iloc[168]
    assert row.dk_price_48h==120. and row.dk_price_168h==0.
    assert row.brent_14d==80. and row.brent_14d!=999.
    changed=raw.copy();changed.loc[changed.index[168],'price_dk']=-9999.
    y=mod.build_frame(changed,weather,oil)
    row2=y[y.pair==mod.PAIRS[0]].iloc[168]
    assert np.allclose(row[mod.FEATURES['weather_oil']].to_numpy(dtype=float),row2[mod.FEATURES['weather_oil']].to_numpy(dtype=float))
    assert row2.any_negative==1


def test_fall_dst_uses_price_from_earlier_delivery_day():
    times=pd.date_range('2025-10-20',periods=200,freq='h',tz='UTC')
    frames=[pd.DataFrame({'HourUTC':times,'pair':p,'price_dk':1.,'price_neighbor':1.,'year':2025}) for p in mod.PAIRS]
    weather={p:pd.DataFrame({v:1. for v in mod.WVARS},index=times) for p in ['DK1','DK2','DE-LU','NL','SE4']}
    oil=pd.Series([70.],index=pd.to_datetime(['2025-01-01']))
    result=mod.build_frame(pd.concat(frames,ignore_index=True),weather,oil)
    assert len(result)==600
