import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import train_dk1_weather_ablation as m


def test_weather_curves_respect_dst_and_missing_repeated_hour():
    times = pd.date_range('2025-10-24','2025-10-28',freq='h',tz='UTC',inclusive='left')
    prices = pd.DataFrame({'p': np.ones(len(times))},index=times)
    weather = pd.DataFrame({v:np.arange(len(times),dtype=float) for v in m.VARIABLES},index=times)
    first_repeated = pd.Timestamp('2025-10-26T00:00Z')
    weather.loc[first_repeated, 'wind_speed_10m'] = np.nan
    curves, coverage = m.weather_curves(prices, lambda zone:weather.copy())
    assert curves.loc['2025-10-26','weather_DK1_wind_speed_10m_h2'] != curves.loc['2025-10-26','weather_DK1_wind_speed_10m_h2']
    assert pd.notna(curves.loc['2025-10-26','weather_DK1_temperature_2m_h2'])
    assert coverage.loc[coverage.zone.eq('DK1') & coverage.variable.eq('wind_speed_10m'),'missing_hours'].iloc[0] == 1


def test_weather_spring_gap_interpolates_clock_hour_only():
    times = pd.date_range('2025-03-28','2025-04-02',freq='h',tz='UTC',inclusive='left')
    prices = pd.DataFrame({'p': np.ones(len(times))},index=times)
    weather = pd.DataFrame({v:np.arange(len(times),dtype=float) for v in m.VARIABLES},index=times)
    curves, _ = m.weather_curves(prices, lambda zone:weather.copy())
    row = curves.loc['2025-03-30']
    assert np.isfinite(row).all()
    assert row['weather_DK1_temperature_2m_h2'] == (row['weather_DK1_temperature_2m_h1'] + row['weather_DK1_temperature_2m_h3']) / 2
