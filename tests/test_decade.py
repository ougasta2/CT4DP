"""Independent checks for the temporal bridge and the four new pilot methods."""
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN

from energy_pilot.decade_data import hourly_prices,solar_bridge,chronological_split
from energy_pilot.decade_methods import CoreAssignment,feature_matrix,stratified_sample,sampled_silhouette
from energy_pilot.decade_evaluation import paired_error_interval,fast_block_interval,cell_predictions
from energy_pilot.evaluation import bootstrap_lift
from energy_pilot.api import SELF,SOLAR


def quarter_fixture():
    return pd.DataFrame({'TimeUTC':pd.date_range('2025-10-01',periods=8,freq='15min',tz='UTC'),
                         'PriceArea':'DK1','DayAheadPriceEUR':[-20.,0.,10.,10.,20.,40.,60.,80.]})


def test_quarters_average_not_sum_and_event_semantics():
    output=hourly_prices(quarter_fixture())
    assert output.SpotPriceEUR.tolist()==[0.,50.]
    assert output.quarter_negative_fraction.tolist()==[.25,0.]
    assert not (output.SpotPriceEUR<0).any()


def test_quarters_require_complete_unique_finite_inputs():
    q=quarter_fixture()
    with pytest.raises(ValueError,match='four'): hourly_prices(q.iloc[:-1])
    with pytest.raises(ValueError,match='Duplicate'): hourly_prices(pd.concat([q,q.iloc[:1]]))
    q.loc[0,'DayAheadPriceEUR']=np.nan
    with pytest.raises(ValueError,match='Missing'): hourly_prices(q)


def test_quarter_dst_has_25_distinct_utc_hours():
    local=pd.date_range('2025-10-26','2025-10-27',tz='Europe/Copenhagen',freq='15min',inclusive='left')
    q=pd.DataFrame({'TimeUTC':local.tz_convert('UTC'),'PriceArea':'DK2','DayAheadPriceEUR':1.})
    assert len(q)==100
    result=hourly_prices(q)
    assert len(result)==25 and result.HourUTC.nunique()==25


def test_solar_definition_break_is_explicit():
    f=pd.DataFrame({'year':[2019,2020],SELF:[30.,5.],'solar_total_est_mwh':[30.,35.],'solar_grid_mwh':[0.,30.]})
    for index,col in enumerate(SOLAR): f[col]=[0.,30. if index==0 else 0.]
    result=solar_bridge(f)
    assert result.solar_reported_total_mwh.tolist()==[30.,35.]
    assert np.isnan(result.solar_grid_mwh.iloc[0])
    assert result.solar_definition.nunique()==2


def test_negative_estimated_self_bound_is_flagged_without_overwriting_source():
    f=pd.DataFrame({'year':[2020],SELF:[-2.],'solar_total_est_mwh':[np.nan],'solar_grid_mwh':[1.]})
    for index,col in enumerate(SOLAR): f[col]=[1. if index==0 else 0.]
    output=solar_bridge(f)
    assert output[SELF].iloc[0]==-2.
    assert output.solar_reported_total_mwh.iloc[0]==-1.
    assert output.solar_harmonized_total_mwh.iloc[0]==1.
    assert output.solar_estimate_floor_applied.iloc[0]


def test_eight_two_split_order_and_no_random_mixing():
    f=pd.DataFrame({'year':range(2016,2026),'HourUTC':pd.to_datetime([f'{year}-01-01T00:00Z' for year in range(2016,2026)])})
    train,test=chronological_split(f)
    assert train.year.tolist()==list(range(2016,2024)) and test.year.tolist()==[2024,2025]
    with pytest.raises(ValueError,match='eight'): chronological_split(f,train_years=[2016],test_years=[2024,2025])
    with pytest.raises(ValueError,match='absent'): chronological_split(f.iloc[1:])


def test_dbscan_extension_preserves_noise_and_freezes_training_cores():
    points=np.array([[0.,0.],[.01,0.],[0.,.01],[5.,5.],[5.01,5.],[5.,5.01]])
    model=DBSCAN(eps=.05,min_samples=2).fit(points)
    extension=CoreAssignment(model)
    predicted=extension.predict(np.array([[0.,.005],[5.005,5.],[100.,100.]]))
    assert predicted.tolist()==[0,1,-1]
    assert len(extension.core_labels)==6


def test_no_core_dbscan_stays_noise():
    model=DBSCAN(eps=.001,min_samples=3).fit([[0.,0.],[1.,1.]])
    assert CoreAssignment(model).predict([[.1,.1]]).tolist()==[-1]


def test_silhouette_sample_without_rare_cluster_is_undefined_not_an_error():
    points=np.arange(5001,dtype=float).reshape(-1,1)
    labels=np.zeros(len(points),dtype=int);labels[-1]=1
    assert np.isnan(sampled_silhouette(points,labels,maximum=10))


def test_supervised_feature_matrix_excludes_price_and_year():
    f=pd.DataFrame({'PriceArea':['DK1','DK2'],'local_hour':[0,23],'month':[1,12],'weekday':[0,6],
                    'wind_to_load':[1.,.5],'solar_to_load':[0.,.3],'load_index':[1.,1.2],
                    'SpotPriceEUR':[100.,-100.],'year':[2016,2025]})
    original=feature_matrix(f)
    f['SpotPriceEUR']=999999;f['year']=3000
    pd.testing.assert_frame_equal(original,feature_matrix(f))
    assert not set(['SpotPriceEUR','year']) & set(original)
    assert feature_matrix(f,False).shape[1]==7 and original.shape[1]==10


def test_dbscan_sampling_represents_every_training_year_area():
    f=pd.DataFrame([{'year':year,'PriceArea':area} for year in range(2016,2024) for area in ['DK1','DK2'] for _ in range(100)])
    selected=stratified_sample(f,maximum=160)
    assert len(selected)==160 and f.iloc[selected].groupby(['year','PriceArea']).size().eq(10).all()
    assert np.array_equal(selected,stratified_sample(f,160))


def test_fast_bootstrap_matches_original_hourly_implementation():
    times=pd.date_range('2024-01-01',periods=40*24,freq='h',tz='UTC')
    frame=pd.DataFrame({'local_date':times.date,'PriceArea':np.where(np.arange(len(times))%2,'DK1','DK2'),
                        'negative_price':(np.arange(len(times))%11)==0})
    baskets=[frozenset(['area='+area,'cluster=C'+str(i%3)]) for i,area in enumerate(frame.PriceArea)]
    antecedent=frozenset(['area=DK1','cluster=C0'])
    a=bootstrap_lift(frame,baskets,antecedent,'negative_price',repeats=30)
    b=fast_block_interval(frame,baskets,antecedent,'negative_price',repeats=30)
    assert a['low']==b['low'] and a['high']==b['high'] and a['valid']==b['valid']


def test_paired_loss_improvement_and_backoff_are_training_only():
    times=pd.date_range('2024-01-01',periods=40*24,freq='h',tz='UTC')
    test=pd.DataFrame({'local_date':times.date})
    result=paired_error_interval(test,np.ones(len(test)),np.zeros(len(test)),repeats=30)
    assert result['improvement']==result['ci_low']==result['ci_high']==1.
    train=pd.DataFrame({'PriceArea':['DK1','DK1'],'season':['winter']*2,'time_of_day':['night']*2,'high_price':[True,False]})
    test=pd.DataFrame({'PriceArea':['DK1'],'season':['summer'],'time_of_day':['morning'],'high_price':[True]})
    probability,count=cell_predictions(train,test,'high_price')
    assert probability.tolist()==[.5] and count==1
