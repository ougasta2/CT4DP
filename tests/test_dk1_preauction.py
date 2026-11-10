import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import train_dk1_preauction as m


def test_vintages_respect_copenhagen_cutoff_and_latest_revision():
    rows = pd.DataFrame([
        ['2025-07-02T12:00:00Z', '2025-07-01T07:00:00Z', 'tso', 'wind', 1],
        ['2025-07-02T12:00:00Z', '2025-07-01T08:00:00Z', 'tso', 'wind', 2],
        ['2025-07-02T12:00:00Z', '2025-07-01T08:00:01Z', 'tso', 'wind', 999],
    ], columns=['target_utc','available_utc','source','variable','value'])
    selected = m.select_vintages(rows, pd.to_datetime(['2025-07-02T12:00:00Z']))
    assert selected.iloc[0, 0] == 2
    assert m.cutoff('2025-01-02') == pd.Timestamp('2025-01-01T09:00Z')


def test_vintage_rejects_naive_and_duplicate_timestamps():
    rows = pd.DataFrame([['2025-07-02T12:00:00Z','2025-07-01T07:00:00Z','tso','wind',1]],
                        columns=['target_utc','available_utc','source','variable','value'])
    with pytest.raises(ValueError, match='duplicate'):
        m.select_vintages(pd.concat([rows, rows]), pd.to_datetime(['2025-07-02T12:00:00Z']))
    rows.loc[0,'available_utc'] = '2025-07-01T07:00:00'
    with pytest.raises(ValueError, match='timezone'):
        m.select_vintages(rows, pd.to_datetime(['2025-07-02T12:00:00Z']))


def test_daily_curves_do_not_use_target_day_prices_and_handle_dst():
    index = pd.date_range('2025-10-20', '2025-10-29', freq='h', tz='UTC', inclusive='left')
    prices = pd.DataFrame({z:np.arange(len(index), dtype=float) for z in m.ZONES}, index=index)
    x, y, data = m.make_daily(prices)
    day = pd.Timestamp('2025-10-26')
    old_x = x.loc[day].copy()
    changed = prices.copy()
    changed.loc[changed.index.tz_convert('Europe/Copenhagen').date == day.date()] += 10000
    new_x, _, _ = m.make_daily(changed)
    pd.testing.assert_series_equal(old_x, new_x.loc[day])
    assert len(data[data.day.eq(day)]) == 25
    assert np.isfinite(y.loc[day]).all()


def test_percentage_denominator_includes_negative_excludes_near_zero():
    result = m.metric([-100, 0, 10, 100], [-96, 100, 100, 106])
    assert result['eligible'] == 2
    assert result['hits_5pct'] == 1
    assert result['rate_5pct'] == .5
