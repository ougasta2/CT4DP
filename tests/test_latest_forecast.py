import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import train_latest_forecast as mod


def test_latest_window_is_365_complete_copenhagen_days():
    start=pd.Timestamp(mod.TEST_START,tz='Europe/Copenhagen').tz_convert('UTC')
    end=(pd.Timestamp(mod.TEST_END,tz='Europe/Copenhagen')+pd.DateOffset(days=1)).tz_convert('UTC')
    assert len(pd.date_range(mod.TEST_START,mod.TEST_END,freq='D'))==365
    assert len(pd.date_range(start,end,freq='h',inclusive='left'))==8760


def test_latest_block_bootstrap_preserves_constant_delta():
    dates=pd.date_range(mod.TEST_START,mod.TEST_END,freq='D').date
    frame=pd.DataFrame({'date':dates})
    assert np.allclose(mod.day_interval(frame,np.full(len(frame),.25),repeats=100),[.25,.25])
