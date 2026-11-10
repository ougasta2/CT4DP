import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('multi_model',Path(__file__).parents[1]/'scripts/train_multiarea.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_training_medians_stay_frozen_and_target_is_conditional():
    f=pd.DataFrame({'HourUTC':pd.to_datetime(['2023-01-01T00:00Z','2024-01-01T00:00Z']),
        'pair':['DK1--NL']*2,'load_dk_mwh':[100.,1000.],'load_neighbor':[200.,2000.],
        'wind_dk_mwh':[20.,100.],'wind_neighbor':[40.,200.],
        'solar_dk_mwh':[0.,0.],'solar_neighbor':[0.,0.],
        'price_dk':[-1.,1.],'price_neighbor':[-2.,-3.]})
    b=mod.FeatureBuilder().fit(f.iloc[:1]);x=b.transform(f)
    assert b.medians['DK1--NL']==(100.,200.)
    assert x.common_load.tolist()==[1.,10.]
    assert x.eligible.tolist()==[True,True] and x.y.tolist()==[0,1]


def test_partial_pool_features_depend_on_correct_pair_only():
    x=np.array([[1.,2.],[3.,4.]])
    z=mod.expanded(x,np.array(['DK1--NL','DK2--SE4']),'partial_pool')
    assert np.allclose(z[0,2:5],[0.,.5,0.])
    assert np.allclose(z[1,2:5],[0.,0.,.5])
    assert np.allclose(z[0,-2:],[0.,0.])


def test_block_interval_keeps_constant_paired_difference():
    dates=pd.date_range('2025-01-01',periods=365,freq='D').date
    f=pd.DataFrame({'year':2025,'date':dates})
    assert np.allclose(mod.block_interval(f,np.ones(len(f)),repeats=100),[1,1])
