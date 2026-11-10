import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import train_dk1_demand_forecast as demand
import train_dk1_demand_price as price


def test_own_demand_inputs_never_use_same_day_or_recent_load():
    idx=pd.date_range('2024-01-01','2024-03-05',freq='h',tz='UTC',inclusive='left')
    loads=pd.Series(np.arange(len(idx),dtype=float)+1000,index=idx)
    weather=pd.DataFrame({k:np.full(len(idx),i+2.) for i,k in enumerate(['temperature_2m','wind_speed_10m','cloud_cover'])},index=idx)
    before,y,_=demand.load_design(loads,weather,'DK1')
    day=pd.Timestamp('2024-03-01')
    local=idx.tz_convert('Europe/Copenhagen').tz_localize(None).normalize()
    changed=loads.copy();changed.loc[local>=day-pd.Timedelta(days=13)] += 10000
    after,_,_=demand.load_design(changed,weather,'DK1')
    pd.testing.assert_series_equal(before.loc[day],after.loc[day])
    assert np.isfinite(before.loc[day]).all()


def test_price_inputs_ignore_true_contemporaneous_demand():
    idx=pd.date_range('2024-01-01','2024-01-04',freq='h',tz='UTC',inclusive='left')
    loc=idx.tz_convert('Europe/Copenhagen')
    frames=[]
    for zone in ['DK1','DK2']:
        frames.append(pd.DataFrame({'HourUTC':idx,'day':loc.tz_localize(None).normalize(),'hour':loc.hour,'zone':zone,
            'forecast':2000.,'tso_backfill':2100.,'load':3000.}))
    source=pd.concat(frames)
    before=price.forecast_curves(idx,source)
    source['load']=999999999
    after=price.forecast_curves(idx,source)
    pd.testing.assert_frame_equal(before['forecast'],after['forecast'])
    pd.testing.assert_frame_equal(before['tso_backfill'],after['tso_backfill'])


def test_price_inputs_reject_duplicate_forecast_times():
    idx=pd.date_range('2024-01-01',periods=24,freq='h',tz='UTC')
    local=idx.tz_convert('Europe/Copenhagen')
    a=pd.DataFrame({'HourUTC':idx,'day':local.tz_localize(None).normalize(),'hour':local.hour,
                    'zone':'DK1','forecast':1.,'tso_backfill':2.})
    with pytest.raises(ValueError,match='duplicate'):
        price.forecast_curves(idx,pd.concat([a,a]))


def test_conditional_interval_optimizer_matches_exhaustive_hit_count():
    import train_dk1_conditional as cond
    rng=np.random.default_rng(123)
    for _ in range(30):
        y=rng.uniform(-180,300,100)
        value=cond.choose(y,75.)
        eligible=y[np.abs(y)>=20]
        candidates=eligible-.05*np.abs(eligible)
        best=max(np.count_nonzero(np.abs(eligible-c)<=.05*np.abs(eligible)+1e-12) for c in candidates)
        assert np.count_nonzero(np.abs(eligible-value)<=.05*np.abs(eligible)+1e-12)==best


def test_vanilla_equation_design_uses_hour_month_temperature_interactions():
    import train_dk1_vanilla_load as vanilla
    idx=pd.DatetimeIndex(['2024-01-01T00:00:00Z','2024-01-01T01:00:00Z'])
    x=vanilla.design(idx,[0.,10.])
    assert x.shape==(2,323)
    # Temperature polynomial is zero at 0 and (1,1,1) at 10 C after scaling.
    assert np.allclose(x[0,212:215],0.)
    assert np.allclose(x[1,212:215],1.)


def test_supply_price_features_ignore_realized_generation():
    import train_dk1_supply_ablation as supply
    idx=pd.date_range('2024-01-01',periods=3,freq='h',tz='UTC')
    base_x=pd.DataFrame({'load_forecast_DK1':1000.,'load_forecast_DK2':800.},index=idx)
    frames=[]
    for zone in ['DK1','DK2']:
        frames.append(pd.DataFrame({'HourUTC':idx,'zone':zone,'forecast_wind_mwh':200.,
            'forecast_solar_harmonized_total_mwh':100.,'wind_mwh':300.}))
    frame=pd.concat(frames);before=supply.supply_features(idx,base_x,frame)
    frame['wind_mwh']=9999999.
    after=supply.supply_features(idx,base_x,frame)
    pd.testing.assert_frame_equal(before,after)
    assert (before.DK1_residual_load==700.).all()


def test_fast_load_inputs_never_use_d_minus_one_or_delivery_day():
    idx=pd.date_range('2024-01-01','2024-02-10',freq='h',tz='UTC',inclusive='left')
    loads=pd.Series(np.arange(len(idx),dtype=float)+1000,index=idx)
    wx=pd.DataFrame({k:np.ones(len(idx)) for k in ['temperature_2m','wind_speed_10m','cloud_cover']},index=idx)
    before,_,_=demand.load_design(loads,wx,'DK1',lags=(2,7,14))
    day=pd.Timestamp('2024-02-01')
    local=idx.tz_convert('Europe/Copenhagen').tz_localize(None).normalize()
    changed=loads.copy();changed.loc[local>=day-pd.Timedelta(days=1)]+=10000
    after,_,_=demand.load_design(changed,wx,'DK1',lags=(2,7,14))
    pd.testing.assert_series_equal(before.loc[day],after.loc[day])


def test_apg_fall_dst_repeated_hours_are_distinct(tmp_path):
    import gzip,json
    import train_dk1_exaa_sample as exaa
    rows=[]
    for label in ['00:00','01:00','2A:00','2B:00']+[f'{h:02d}:00' for h in range(3,24)]:
        rows.append({'TF':label,'V':[{'V':42.,'M':False}]})
    path=tmp_path/'2024-10-27.json.gz'
    with gzip.open(path,'wt',encoding='utf8') as f:json.dump({'body':{'ResponseData':{'ValueColumns':[{'InternalName':'EXAAD1Price'}],'ValueRows':rows}}},f)
    values=exaa.read_apg(path)
    assert len(values)==25 and not values.index.has_duplicates
    assert (values.index[3]-values.index[2])==pd.Timedelta(hours=1)


def test_hit_probability_expansion_has_fixed_candidates_and_no_target_leak():
    import train_dk1_hit_probability as model
    x=pd.DataFrame({'a':[1.,2.]});ref=np.array([100.,200.])
    a,points,labels=model.expand(x,ref)
    assert labels is None and a.shape==(26,2) and len(points)==26
    _,points2,labels2=model.expand(x,ref,np.array([110.,180.]))
    np.testing.assert_array_equal(points,points2)
    assert labels2.dtype==bool and len(labels2)==26


def test_capacity_rejects_incomplete_quarter_hour_coverage(tmp_path, monkeypatch):
    import gzip, json
    import train_dk1_capacity_ablation as model
    rows = [{'ts_utc': f'2026-09-01T00:{m:02d}:00Z', 'ntc_mw': 100 + m}
            for m in [0, 15, 30, 45]]
    rows += [{'ts_utc': f'2026-09-01T01:{m:02d}:00Z', 'ntc_mw': 200}
             for m in [0, 15, 30]]
    rows += [{'ts_utc': '2026-09-01T02:15:00Z', 'ntc_mw': 300}]
    path = tmp_path/'DK_1_DE_LU_2026-09.json.gz'
    with gzip.open(path, 'wt', encoding='utf8') as f:
        json.dump({'body': {'data': rows}}, f)
    monkeypatch.setattr(model, 'ROOT', tmp_path)
    values = model.capacity('DK_1', 'DE_LU')
    assert values.iloc[0] == 122.5
    assert values.iloc[1:].isna().all()
