"""Exploratory 11:00 German EXAA + cross-border capacity ablation.

Capacity is a later ENTSO-E backfill, not a point-in-time archive. Its
publication and revision history must be checked before operational use.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import train_dk1_exaa_de as exaa
import train_dk1_nonlinear as nonlinear
import train_dk1_preauction as base
from train_dk1_exaa_sample import paired_interval

ROOT = Path('data/external/entsoe_ntc_backfill')
OUT = Path('results/dk1_exaa_de')


def capacity(origin, destination, years=(2026,)):
    frames = []
    for year in years:
        for path in sorted(ROOT.glob(f'{origin}_{destination}_{year}-*.json.gz')):
            with gzip.open(path, 'rt', encoding='utf8') as handle:
                body = json.load(handle)['body']
            if body.get('truncated'):
                raise ValueError(f'Truncated capacity source: {path}')
            frames.extend(body['data'])
    if not frames:
        raise ValueError(f'No capacity data for {origin} to {destination}')
    df = pd.DataFrame(frames)
    df['ts_utc'] = pd.to_datetime(df.ts_utc, utc=True)
    if df.ts_utc.duplicated().any():
        raise ValueError('Duplicate capacity timestamp')
    df = df.set_index('ts_utc').sort_index()
    # Permit one hourly value or four quarter-hour values, but not partial hours.
    df = df[df.ntc_mw.notna()].copy()
    df['minute'] = df.index.minute
    grouped = df.resample('h')
    minutes = grouped.minute.agg(lambda values: tuple(sorted(values)))
    complete = minutes.map(lambda values: values in [(0,), (0, 15, 30, 45)])
    return grouped.ntc_mw.mean().where(complete)


def run():
    p = pd.read_csv('results/dk1_demand_price/evaluation_2026_predictions.csv.gz',
                    parse_dates=['HourUTC', 'day']).set_index('HourUTC').sort_index()
    x = nonlinear.design(p)
    x['EXAA_DE'] = exaa.load_exaa('DE_LU').reindex(p.index)
    x['EXAA_AT'] = exaa.load_exaa('AT').reindex(p.index)
    x['DE_AT_spread'] = x.EXAA_DE - x.EXAA_AT
    x['DE_previous_spread'] = x.EXAA_DE - x.DK1_lag1
    x['hgb_price'] = p.hgb_price
    x['ntc_export'] = capacity('DK_1', 'DE_LU').reindex(p.index)
    x['ntc_import'] = capacity('DE_LU', 'DK_1').reindex(p.index)
    valid = x.notna().all(axis=1)
    x, p = x.loc[valid], p.loc[valid]
    pieces = []
    for month in pd.date_range('2026-08-01', '2026-11-01', freq='MS', inclusive='left'):
        train = p.day.lt(month)
        test = p.day.ge(month) & p.day.lt(month + pd.offsets.MonthBegin(1))
        if train.sum() < 250 or not test.any():
            continue
        out = p.loc[test, ['day', 'hour', 'DK1']].copy()
        out['raw_de'] = x.loc[test, 'EXAA_DE']
        out['spread_0.1'] = out.raw_de + .1 * (x.loc[test, 'DK1_lag1'] - x.loc[test, 'DE-LU_lag1'])
        residual = (p.DK1 - x.EXAA_DE).loc[train]
        for name, columns in [('without_ntc', x.columns.drop(['ntc_export', 'ntc_import'])),
                              ('with_ntc', x.columns)]:
            model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=80,
                learning_rate=.04, max_leaf_nodes=7, min_samples_leaf=40,
                l2_regularization=10, early_stopping=False, random_state=42)
            model.fit(x.loc[train, columns], residual)
            out[name] = out.raw_de + model.predict(x.loc[test, columns])
        pieces.append(out)
        print(month.date(), 'train', train.sum(), 'test', test.sum(), flush=True)
    if len(pieces) != 3:
        raise ValueError('Incomplete monthly evaluation; finish downloading capacity first')
    out = pd.concat(pieces).sort_index()
    models = ['raw_de', 'spread_0.1', 'without_ntc', 'with_ntc']
    metrics = []
    for label, group in list(out.groupby(out.day.dt.strftime('%Y-%m'))) + [
            ('2026-09-to-10', out[out.day.ge('2026-09-01')])]:
        for name in models:
            metrics.append({'period': label, 'model': name, **base.metric(group.DK1, group[name])})
    table = pd.DataFrame(metrics)
    table.to_csv(OUT / 'capacity_metrics.csv', index=False)
    test = out[out.day.ge('2026-09-01')]
    bounds = paired_interval(test, 'with_ntc', baseline='without_ntc')
    pd.DataFrame([{'period': '2026-09-to-10', 'model': 'with_ntc',
        'baseline': 'without_ntc', 'mae_improvement_low': bounds[0, 0],
        'mae_improvement_high': bounds[1, 0], 'hit_improvement_low': bounds[0, 1],
        'hit_improvement_high': bounds[1, 1], 'block_days': 7}]).to_csv(
            OUT / 'capacity_paired_interval.csv', index=False)
    (OUT / 'capacity_protocol.json').write_text(json.dumps({
        'issue': '11:00 D-1 Europe/Copenhagen; EXAA and NTC publication assumed, not audited',
        'source': 'https://entsoemcp.com/api/series; ENTSO-E revised NTC backfill',
        'method': 'monthly residual HGB, same rows and configuration with/without two directional capacities',
        'training': 'previous 2026 hours only',
        'caveats': ['source may include revisions made after bidding', '2026 already examined; exploratory',
                    '7-day block bootstrap does not adjust for model searches',
                    'EXAA availability is not verified for each day'],
    }, indent=2) + '\n', encoding='utf8')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
