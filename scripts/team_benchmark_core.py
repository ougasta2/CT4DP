"""Common evaluation for the main workspace and standalone team repositories."""

import json
import time

import numpy as np
import pandas as pd


def metric(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    err = np.abs(y-pred)
    eligible = np.abs(y) >= 20
    return {'hours': len(y), 'mae': float(err.mean()), 'rmse': float(np.sqrt(np.mean((y-pred)**2))),
            'eligible': int(eligible.sum()), 'hits_5pct': int((err[eligible] <= .05*np.abs(y[eligible])).sum()),
            'rate_5pct': float((err[eligible] <= .05*np.abs(y[eligible])).mean()) if eligible.any() else None,
            'negative_hours': int((y < 0).sum()), 'negative_mae': float(err[y < 0].mean()) if (y < 0).any() else None}


def run_prepared(name, build, start, end, x, y, hourly, dest, min_train_days=365):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.day != 1 or end.day != 1 or end <= start:
        raise ValueError('start and end must be first-of-month dates, with end after start')
    predictions, timings = [], []
    for month in pd.date_range(start, end, freq='MS', inclusive='left'):
        next_month = month + pd.offsets.MonthBegin(1)
        train = x.index[x.index < month]
        evaluate = x.index[(x.index >= month) & (x.index < next_month)]
        if len(train) < min_train_days or len(evaluate) != (next_month-month).days:
            raise ValueError(f'Insufficient training or incomplete evaluation days for {month:%Y-%m}')
        model = build()
        start_time = time.perf_counter()
        model.fit(x.loc[train], y.loc[train])
        fit_seconds = time.perf_counter() - start_time
        start_time = time.perf_counter()
        curves = np.asarray(model.predict(x.loc[evaluate]))
        predict_seconds = time.perf_counter() - start_time
        if curves.shape != (len(evaluate), 24) or not np.isfinite(curves).all():
            raise ValueError(f'{name}: expected finite (days, 24) predictions')
        target = hourly[hourly.day.isin(evaluate)][['day', 'hour', 'DK1']].copy()
        target['prediction'] = pd.DataFrame(curves, index=evaluate).loc[pd.DatetimeIndex(target.day)].to_numpy()[np.arange(len(target)), target.hour.to_numpy()]
        target.index.name = 'HourUTC'
        target = target.rename(columns={'DK1':'actual'})
        predictions.append(target)
        timings.append({'month': str(month.date()), 'train_days': len(train), 'evaluation_days': len(evaluate),
                        'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds})
        print(f'{name}: {month:%Y-%m}, {len(train)} training days', flush=True)
    pred = pd.concat(predictions).sort_index()
    if pred.index.has_duplicates or pred.isna().any().any():
        raise ValueError('Duplicate timestamps or missing output values')
    dest.mkdir(parents=True, exist_ok=True)
    pred.to_csv(dest / 'predictions.csv.gz', compression={'method':'gzip', 'mtime':0})
    pd.DataFrame(timings).to_csv(dest / 'timing.csv', index=False)
    metrics = [{'year': int(year), **metric(group.actual, group.prediction)}
               for year, group in pred.groupby(pred.day.dt.year)]
    pd.DataFrame(metrics).to_csv(dest / 'metrics.csv', index=False)
    (dest / 'run.json').write_text(json.dumps({'model':name, 'start':str(start.date()), 'end_exclusive':str(end.date()),
        'training':'expanding, monthly refit', 'cutoff':'10:00 Europe/Copenhagen on D-1',
        'inputs':'lagged five-zone prices, calendar, archived weather',
        'warning':'Historical source release times and revisions are not fully audited; evaluation years have been inspected.'}, indent=2)+'\n', encoding='utf-8')
    return pred
