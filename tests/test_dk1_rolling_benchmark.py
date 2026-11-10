import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import train_dk1_rolling_benchmark as m


def test_hit_optimized_prediction_uses_actual_sample_denominators():
    samples=np.array([100.]*20+[150.]*5+[1.]*20)
    p=m.best_interval_prediction(samples,101.)
    assert np.sum(np.abs(samples-p)<=.05*np.abs(samples)) == 20


def test_distribution_calibration_never_uses_future_residuals():
    idx=pd.date_range('2022-01-01', periods=70, freq='D',tz='UTC')
    df=pd.DataFrame({'day':idx.tz_localize(None),'hour':12,'DK1':np.arange(70,dtype=float)+60.,'lear_ensemble':100.},index=idx)
    first=m.apply_distribution(df)
    changed=df.copy();changed.iloc[-1,changed.columns.get_loc('DK1')]=99999
    second=m.apply_distribution(changed)
    np.testing.assert_allclose(first.interval_point.iloc[:69],second.interval_point.iloc[:69],equal_nan=True)
    assert np.isnan(first.interval_point.iloc[29]) and np.isfinite(first.interval_point.iloc[30])


def test_future_prices_never_affect_extended_previous_price_inputs():
    idx=pd.date_range('2022-10-20','2022-11-03',freq='h',tz='UTC',inclusive='left')
    prices=pd.DataFrame({zone:np.arange(len(idx),dtype=float) for zone in m.base.ZONES},index=idx)
    orig,_,_=m.base.make_daily(prices)
    baseline=m.price_features(prices,orig)
    day=pd.Timestamp('2022-10-30')
    altered=prices.copy()
    altered.loc[altered.index.tz_convert('Europe/Copenhagen').date>=day.date()] += 10000
    orig2,_,_=m.base.make_daily(altered)
    after=m.price_features(altered,orig2)
    pd.testing.assert_series_equal(baseline.loc[day],after.loc[day])


def test_online_bias_does_not_leak_with_repeated_dst_hour():
    import evaluate_dk1_calibration as calibration
    idx=pd.date_range('2024-09-01','2024-10-28',freq='h',tz='UTC',inclusive='left')
    local=idx.tz_convert('Europe/Copenhagen')
    p=pd.DataFrame({'day':local.tz_localize(None).normalize(),'hour':local.hour,'DK1':100.,'ridge_original_weather':90.,'lear_raw_3y':95.},index=idx)
    result=calibration.add_corrections(p)
    changed=p.copy();changed.loc[pd.Timestamp('2024-10-27T00:00Z'),'DK1']=9999
    after=calibration.add_corrections(changed)
    before=result.loc['2024-10-27T01:00Z','lear_raw_3y_median30']
    assert before==after.loc['2024-10-27T01:00Z','lear_raw_3y_median30']


def test_tso_forecast_curve_drops_missing_repeated_hour():
    import train_dk1_renewable_ablation as renewable
    idx=pd.date_range('2024-10-26T22:00Z','2024-10-27T23:00Z',freq='h',inclusive='left')
    values=pd.DataFrame({'DK1_Solar':np.arange(len(idx),dtype=float)},index=idx)
    values.loc[pd.Timestamp('2024-10-27T00:00Z'),'DK1_Solar']=np.nan
    day=renewable.curves(values)
    assert day.loc[pd.Timestamp('2024-10-27')].isna().all()
