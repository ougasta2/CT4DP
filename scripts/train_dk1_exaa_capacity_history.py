"""Full APG history, with/without revised DK1-DE capacity (exploratory)."""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import train_dk1_capacity_ablation as ntc
import train_dk1_exaa_sample as apg
import train_dk1_nonlinear as nonlinear
import train_dk1_preauction as base

OUT = Path('results/dk1_exaa_full')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv('results/dk1_demand_price/nonlinear_predictions.csv.gz',
                    parse_dates=['HourUTC', 'day']).set_index('HourUTC').sort_index()
    x = nonlinear.design(p)
    x['hgb_price'] = p.hgb_price
    x['exaa_at'] = apg.source(full=True).reindex(p.index)
    x['at_previous_spread'] = x.exaa_at - x.DK1_lag1
    x['ntc_export'] = ntc.capacity('DK_1', 'DE_LU', years=(2023, 2024, 2025)).reindex(p.index)
    x['ntc_import'] = ntc.capacity('DE_LU', 'DK_1', years=(2023, 2024, 2025)).reindex(p.index)
    valid = x.notna().all(axis=1)
    p, x = p.loc[valid], x.loc[valid]
    pieces = []
    for month in pd.date_range('2024-01-01', '2026-01-01', freq='MS', inclusive='left'):
        train = p.day.lt(month)
        test = p.day.ge(month) & p.day.lt(month + pd.offsets.MonthBegin(1))
        if train.sum() < 1000 or not test.any():
            raise ValueError(f'Insufficient training or evaluation data for {month}')
        out = p.loc[test, ['day', 'hour', 'DK1', 'hgb_price']].copy()
        out['blend_half'] = .5 * (out.hgb_price + x.loc[test, 'exaa_at'])
        for name, columns in [('without_capacity', x.columns.drop(['ntc_export', 'ntc_import'])),
                              ('with_capacity', x.columns)]:
            model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=100,
                learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=40,
                l2_regularization=10, early_stopping=False, random_state=42)
            model.fit(x.loc[train, columns], p.loc[train, 'DK1'])
            out[name] = model.predict(x.loc[test, columns])
        pieces.append(out)
        print(month.date(), 'train', train.sum(), 'test', test.sum(), flush=True)
    out = pd.concat(pieces).sort_index()
    models = ['hgb_price', 'blend_half', 'without_capacity', 'with_capacity']
    rows, intervals = [], []
    for year, group in out.groupby(out.day.dt.year):
        for model in models:
            rows.append({'year': year, 'days': group.day.nunique(), 'model': model,
                         **base.metric(group.DK1, group[model])})
        bounds = apg.paired_interval(group, 'with_capacity', baseline='without_capacity')
        intervals.append({'year': year, 'mae_improvement_low': bounds[0, 0],
                          'mae_improvement_high': bounds[1, 0],
                          'hit_improvement_low': bounds[0, 1],
                          'hit_improvement_high': bounds[1, 1], 'block_days': 7})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'capacity_history_metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(OUT / 'capacity_history_intervals.csv', index=False)
    (OUT / 'capacity_history_protocol.json').write_text(json.dumps({
        'cutoff': '11:00 D-1 Copenhagen; publication and revisions not individually audited',
        'inputs': 'APG EXAA Austria plus both directional ENTSO-E NTC backfills',
        'training': 'monthly, previous dates only; matched rows across models',
        'caveats': ['2024 and 2025 previously examined; exploratory',
                    '2026 German EXAA experiment informed trying capacity',
                    'capacity backfills may include post-auction revisions',
                    'intervals do not adjust for prior model searches'],
    }, indent=2) + '\n', encoding='utf8')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
