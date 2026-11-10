import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import train_price_regression as model


def test_relative_error_only_qualifies_away_from_zero():
    r=model.score(np.array([-100.,0.,10.,100.]),np.array([-98.,100.,10.,104.]))
    assert r['percent3_eligible']==2 and r['percent3_hits']==1 and r['percent3_rate']==.5


def test_previous_local_day_is_not_same_day_during_fall_dst():
    times=pd.date_range('2025-10-24','2025-10-28',freq='h',tz='UTC',inclusive='left')
    f=pd.DataFrame({'HourUTC':times,'price':np.arange(len(times),dtype=float)})
    lag=model.local_lag(f,1)
    local=f.HourUTC.dt.tz_convert('Europe/Copenhagen')
    for i in np.flatnonzero(np.isfinite(lag)):
        assert lag[i] < f.price.iloc[i]
    assert local.dt.date.nunique()>=4
