"""UTC keys, coverage audits, complete-case cleaning, and train-only features."""
import numpy as np
import pandas as pd

from .api import KEYS, WIND, SOLAR, SELF

TZ = 'Europe/Copenhagen'
FEATURES = ['wind_to_load', 'solar_to_load', 'load_index']


def keyed(frame):
    frame = frame.copy()
    frame['HourUTC'] = pd.to_datetime(frame.HourUTC, utc=True, errors='raise')
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError('Null or duplicate (HourUTC, PriceArea) keys')
    if not frame.PriceArea.isin(['DK1', 'DK2']).all():
        raise ValueError('Unexpected price area')
    if not (frame.HourUTC == frame.HourUTC.dt.floor('h')).all():
        raise ValueError('Non-hourly timestamp')
    return frame


def coverage(frame, start, end, source):
    frame = keyed(frame)
    expected_hours = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end, tz=TZ),
                                   freq='h', inclusive='left').tz_convert('UTC')
    expected = pd.MultiIndex.from_product([expected_hours, ['DK1', 'DK2']], names=KEYS)
    actual = pd.MultiIndex.from_frame(frame[KEYS])
    if len(actual.difference(expected)):
        raise ValueError(f'{source}: rows outside requested local-time window')
    rows = []
    for area in ['DK1', 'DK2']:
        expected_area = pd.DataFrame({'HourUTC': expected_hours})
        expected_area['month'] = expected_area.HourUTC.dt.tz_convert(TZ).dt.strftime('%Y-%m')
        f = frame[frame.PriceArea == area].copy()
        f['month'] = f.HourUTC.dt.tz_convert(TZ).dt.strftime('%Y-%m')
        for month, n in expected_area.groupby('month').size().items():
            subset = f[f.month == month]
            rows.append(dict(source=source, area=area, month=month, expected=int(n),
                             observed=len(subset), missing=int(n-len(subset)),
                             null_cells=int(subset.isna().sum().sum())))
    return pd.DataFrame(rows)


def clean_join(production, prices):
    production, prices = keyed(production), keyed(prices)
    merged = production.merge(prices, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    audit = {'production_rows': len(production), 'price_rows': len(prices),
             'merge_counts': {str(k): int(v) for k, v in merged._merge.value_counts().items()}}
    audit['missing_fraction_by_column'] = merged.drop(columns='_merge').isna().mean().to_dict()
    frame = merged[merged._merge == 'both'].drop(columns='_merge').copy()
    numeric = WIND + SOLAR + [SELF, 'GrossConsumptionMWh', 'SpotPriceEUR']
    for c in numeric:
        frame[c] = pd.to_numeric(frame[c], errors='raise')
    audit['negative_physical_cells'] = int((frame[WIND+SOLAR+[SELF, 'GrossConsumptionMWh']] < 0).sum().sum())
    valid = np.isfinite(frame[WIND+SOLAR+['GrossConsumptionMWh', 'SpotPriceEUR']]).all(axis=1)
    valid &= (frame[WIND+SOLAR] >= 0).all(axis=1) & (frame.GrossConsumptionMWh > 0)
    audit['excluded_incomplete_or_invalid_rows'] = int((~valid).sum())
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError(f'No usable joined rows: {audit}')
    frame['wind_mwh'] = frame[WIND].sum(axis=1, min_count=len(WIND))
    frame['solar_grid_mwh'] = frame[SOLAR].sum(axis=1, min_count=len(SOLAR))
    frame['solar_total_est_mwh'] = frame.solar_grid_mwh + frame[SELF].where(frame[SELF] >= 0)
    frame['gross_load_mwh'] = frame.GrossConsumptionMWh
    local = frame.HourUTC.dt.tz_convert(TZ)
    frame['year'], frame['month'], frame['local_hour'] = local.dt.year, local.dt.month, local.dt.hour
    frame['weekday'] = local.dt.weekday
    frame['local_date'] = local.dt.date
    frame['season'] = frame.month.map({12:'winter',1:'winter',2:'winter',3:'spring',4:'spring',5:'spring',
                                       6:'summer',7:'summer',8:'summer',9:'autumn',10:'autumn',11:'autumn'})
    frame['time_of_day'] = pd.cut(frame.local_hour, [-1,5,11,17,23],
                                  labels=['night','morning','afternoon','evening']).astype(str)
    audit['usable_rows'] = len(frame)
    audit['join_retention_fraction'] = len(frame) / max(len(production), len(prices))
    return frame.sort_values(KEYS).reset_index(drop=True), audit


class Features:
    def fit(self, train):
        if train.empty or not (train.gross_load_mwh > 0).all():
            raise ValueError('Training requires nonempty, strictly positive load')
        self.medians = train.groupby('PriceArea').gross_load_mwh.median().to_dict()
        self.quantiles = {a: g.gross_load_mwh.quantile([1/3,2/3]).to_numpy()
                          for a,g in train.groupby('PriceArea')}
        self.thresholds = train.groupby('PriceArea').SpotPriceEUR.quantile(.95).to_dict()
        return self

    def transform(self, frame, solar_total=False):
        frame = frame.copy()
        if set(frame.PriceArea) - set(self.medians):
            raise ValueError('Validation contains an area absent from training')
        frame['wind_to_load'] = frame.wind_mwh / frame.gross_load_mwh
        frame['solar_to_load'] = frame['solar_total_est_mwh' if solar_total else 'solar_grid_mwh'] / frame.gross_load_mwh
        frame['load_index'] = frame.gross_load_mwh / frame.PriceArea.map(self.medians)
        frame['demand'] = [ ['low','mid','high'][np.searchsorted(self.quantiles[a], x, side='left')]
                           for a,x in zip(frame.PriceArea, frame.gross_load_mwh)]
        frame['negative_price'] = frame.SpotPriceEUR < 0
        frame['high_price'] = frame.SpotPriceEUR > frame.PriceArea.map(self.thresholds)
        return frame
