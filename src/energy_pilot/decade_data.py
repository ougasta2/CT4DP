"""Audited 2016–2025 inputs, explicit solar and market-resolution bridges."""
import numpy as np
import pandas as pd

from .api import Client, PRODUCTION, PRICES, QUARTER_PRICES, KEYS, WIND, SOLAR, SELF
from .data import TZ, keyed, coverage, clean_join

YEARS=list(range(2016,2026))
TRAIN_YEARS=YEARS[:8]
TEST_YEARS=YEARS[8:]


def hourly_prices(quarter):
    """Four equal-duration quarters -> mean EUR/MWh, never a sum or partial mean.

    Price events after the transition are events in this hourly mean, not events
    in any individual quarter. Preserve quarter negativity for sensitivity audit.
    """
    frame=quarter.copy()
    frame['TimeUTC']=pd.to_datetime(frame.TimeUTC,utc=True,errors='raise')
    if frame.duplicated(['TimeUTC','PriceArea']).any():
        raise ValueError('Duplicate quarter-hour price keys')
    if frame[['TimeUTC','PriceArea','DayAheadPriceEUR']].isna().any().any():
        raise ValueError('Missing quarter-hour prices or keys')
    if not frame.PriceArea.isin(['DK1','DK2']).all():
        raise ValueError('Unexpected quarter-hour price area')
    if not np.isfinite(frame.DayAheadPriceEUR).all():
        raise ValueError('Nonfinite quarter-hour prices')
    if not frame.TimeUTC.eq(frame.TimeUTC.dt.floor('15min')).all():
        raise ValueError('Non-quarter-hour price timestamp')
    frame['HourUTC']=frame.TimeUTC.dt.floor('h')
    frame['quarter_negative_fraction']=(frame.DayAheadPriceEUR<0).astype(float)
    group=frame.groupby(KEYS,sort=True)
    if not group.size().eq(4).all():
        raise ValueError('Each hourly average requires all four distinct quarters')
    result=group.agg(SpotPriceEUR=('DayAheadPriceEUR','mean'),
                     quarter_negative_fraction=('quarter_negative_fraction','mean')).reset_index()
    result['price_resolution']='mean_of_4_quarters'
    return keyed(result)


def solar_bridge(frame):
    """Make the documented pre-2020 solar layout visible, without zero imputation."""
    frame=frame.copy()
    before=frame.year<2020
    # The original metadata places ALL solar in SELF before 2020, not just self-use.
    # Keep the unmodified signed source total, including documented anomalies.
    frame['solar_reported_total_mwh']=frame[SOLAR].sum(axis=1,min_count=len(SOLAR))+frame[SELF]
    frame.loc[before,'solar_reported_total_mwh']=frame.loc[before,SELF]
    # Current official metadata defines estimated self-use as max(model-actual,0).
    # 2020 historical records violate that bound. Apply only this stated bound,
    # retain the original column and mark every correction; never impute nulls.
    frame['solar_estimate_floor_applied']=(~before)&(frame[SELF]<0)
    frame['solar_harmonized_total_mwh']=frame[SOLAR].sum(axis=1,min_count=len(SOLAR))+frame[SELF].clip(lower=0)
    frame.loc[before,'solar_harmonized_total_mwh']=frame.loc[before,SELF]
    frame.loc[before,'solar_grid_mwh']=np.nan  # no separately observed grid series
    frame['solar_definition']=np.where(before,'all_solar_in_legacy_self_column',
                                       'grid_plus_estimated_self_consumption')
    frame['solar_total_est_mwh']=frame.solar_harmonized_total_mwh
    valid=np.isfinite(frame.solar_harmonized_total_mwh)&(frame.solar_harmonized_total_mwh>=0)
    if not valid.all():
        raise ValueError(f'{int((~valid).sum())} rows lack the required total-solar proxy')
    return frame


def load_decade(client):
    """Bounded bulk windows avoid >5-minute discontinued-API waits per year."""
    for dataset in [PRODUCTION,PRICES,QUARTER_PRICES]:
        client.metadata(dataset)
    production=pd.concat([
        client.window(PRODUCTION,'2016-01-01','2023-01-01',limit=150000),
        client.year(PRODUCTION,2023),client.year(PRODUCTION,2024),client.year(PRODUCTION,2025)
    ],ignore_index=True)
    historical=pd.concat([
        client.window(PRICES,'2016-01-01','2023-01-01',limit=150000),
        client.year(PRICES,2023),client.year(PRICES,2024),
        client.window(PRICES,'2025-01-01','2025-10-01')
    ],ignore_index=True)
    historical['price_resolution']='original_hourly'
    historical['quarter_negative_fraction']=np.nan
    quarter=hourly_prices(client.window(QUARTER_PRICES,'2025-10-01','2026-01-01'))
    prices=keyed(pd.concat([historical,quarter],ignore_index=True))
    audits=[]
    for source,frame in [(PRODUCTION,production),('hourly_price_bridge',prices)]:
        local_year=pd.to_datetime(frame.HourUTC,utc=True).dt.tz_convert(TZ).dt.year
        for year in YEARS:
            observed=frame[local_year==year]
            if source=='hourly_price_bridge':
                observed=observed[KEYS+['SpotPriceEUR']]
            audits.append(coverage(observed,f'{year}-01-01',f'{year+1}-01-01',source))
    audit=pd.concat(audits,ignore_index=True)
    if audit.missing.sum():
        missing=audit[audit.missing!=0].to_dict('records')
        raise ValueError(f'Incomplete decade: {missing}')
    frame,merge=clean_join(production,prices)
    # Old solar-grid components are structurally zero in this source, not absent
    # observations of total solar. clean_join never fills missing numeric values.
    frame=solar_bridge(frame)
    usable=frame.groupby(['year','PriceArea']).size().rename('usable').reset_index()
    expected=audit[audit.source==PRODUCTION].assign(year=lambda x:x.month.str[:4].astype(int)).groupby(
        ['year','area']).expected.sum().rename('expected').reset_index().rename(columns={'area':'PriceArea'})
    usable=expected.merge(usable,on=['year','PriceArea'],how='left')
    usable['retention']=usable.usable/usable.expected
    if usable.retention.min()<.995:
        raise ValueError(f'Usable coverage below 99.5%: {usable.to_dict("records")}')
    return frame,audit,merge,usable


def chronological_split(frame,train_years=TRAIN_YEARS,test_years=TEST_YEARS):
    if len(train_years)!=8 or len(test_years)!=2 or max(train_years)>=min(test_years):
        raise ValueError('Require eight older training years and two strictly later test years')
    if sorted(list(train_years)+list(test_years))!=YEARS:
        raise ValueError('The latest ten completed calendar years are 2016–2025 in this snapshot')
    train=frame[frame.year.isin(train_years)].copy().reset_index(drop=True)
    test=frame[frame.year.isin(test_years)].copy().reset_index(drop=True)
    if set(train.year)!=set(train_years) or set(test.year)!=set(test_years):
        raise ValueError('A required training/test year is absent')
    if train.HourUTC.max()>=test.HourUTC.min():
        raise ValueError('Temporal split overlaps')
    return train,test
