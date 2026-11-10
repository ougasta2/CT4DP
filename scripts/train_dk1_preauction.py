"""DK1 day-ahead price experiments with a 10:00 D-1 information cutoff.

Optional forecast CSV columns: target_utc, available_utc, source, variable, value.
Each row must represent a retained historical vintage, not a current revision.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import train_latest_forecast as old

OUT = Path('results/dk1_preauction')
ZONES = ['DK1', 'DK2', 'DE-LU', 'NL', 'SE4']
DEV = pd.Timestamp('2024-01-01')
TEST = pd.Timestamp('2025-11-10')
END = pd.Timestamp('2026-11-10')


def cutoff(day):
    return (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=10)).tz_localize('Europe/Copenhagen').tz_convert('UTC')


def select_vintages(records, targets):
    """Latest value actually available before cutoff; reject ambiguous records."""
    r = records.copy()
    required = {'target_utc', 'available_utc', 'source', 'variable', 'value'}
    if not required.issubset(r.columns):
        raise ValueError('forecast CSV must contain ' + ', '.join(sorted(required)))
    for c in ['target_utc', 'available_utc']:
        # Naive timestamps are ambiguous and must never silently become UTC.
        if not r[c].astype(str).str.contains(r'(?:Z|[+-]\d\d:\d\d)$', regex=True).all():
            raise ValueError(f'{c} must include timezone offset')
        r[c] = pd.to_datetime(r[c], utc=True, errors='raise')
    if r[['source', 'variable', 'value']].isna().any().any() or not np.isfinite(pd.to_numeric(r.value, errors='raise')).all():
        raise ValueError('invalid forecast values or provenance')
    if r.duplicated(['target_utc', 'available_utc', 'source', 'variable']).any():
        raise ValueError('duplicate forecast vintage')
    dates = r.target_utc.dt.tz_convert('Europe/Copenhagen').dt.tz_localize(None).dt.normalize()
    limits = dates.map(cutoff)
    r = r[r.available_utc <= limits].sort_values('available_utc')
    r = r.drop_duplicates(['target_utc', 'source', 'variable'], keep='last')
    r['feature'] = r.source.astype(str) + '__' + r.variable.astype(str)
    p = r.pivot(index='target_utc', columns='feature', values='value')
    return p.reindex(pd.DatetimeIndex(targets))


def load_prices():
    raw = old.price_pairs()
    frames = {}
    for zone in ZONES:
        candidates = []
        for pair, g in raw.groupby('pair'):
            a, b = pair.split('--')
            if zone == a: candidates.append(g[['HourUTC', 'price_dk']].rename(columns={'price_dk':'price'}))
            if zone == b: candidates.append(g[['HourUTC', 'price_neighbor']].rename(columns={'price_neighbor':'price'}))
        x = pd.concat(candidates)
        if (x.groupby('HourUTC').price.nunique() > 1).any():
            raise ValueError('conflicting prices: ' + zone)
        frames[zone] = x.drop_duplicates('HourUTC').set_index('HourUTC').price.sort_index()
    p = pd.DataFrame(frames).dropna()
    if not p.index.equals(pd.date_range(p.index.min(), p.index.max(), freq='h')):
        raise ValueError('missing hourly prices')
    return p


def make_daily(prices):
    local = prices.index.tz_convert('Europe/Copenhagen')
    data = prices.copy()
    data['day'] = local.tz_localize(None).normalize()
    data['hour'] = local.hour
    # Fall repeated hour is averaged; spring missing hour interpolated for model inputs/targets.
    days = pd.DatetimeIndex(sorted(data.day.unique()))
    curves = {}
    for z in ZONES:
        v = data.pivot_table(index='day', columns='hour', values=z, aggfunc='mean').reindex(index=days, columns=range(24))
        curves[z] = v.T.interpolate(axis=0, limit_direction='both').T
    features = []
    for z in ZONES:
        for lag in [1, 7]:
            x = curves[z].shift(lag).copy()
            x.columns = [f'{z}_lag{lag}_h{h}' for h in range(24)]
            features.append(x)
    x = pd.concat(features, axis=1)
    x['weekday_sin'] = np.sin(2*np.pi*days.dayofweek/7)
    x['weekday_cos'] = np.cos(2*np.pi*days.dayofweek/7)
    x['year_sin'] = np.sin(2*np.pi*days.dayofyear/365.25)
    x['year_cos'] = np.cos(2*np.pi*days.dayofyear/365.25)
    return x, curves['DK1'], data


def metric(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    err = np.abs(y-pred)
    eligible = np.abs(y) >= 20
    return {'hours': len(y), 'mae': float(err.mean()), 'rmse': float(np.sqrt(np.mean((y-pred)**2))),
            'eligible': int(eligible.sum()), 'hits_5pct': int((err[eligible] <= .05*np.abs(y[eligible])).sum()),
            'rate_5pct': float((err[eligible] <= .05*np.abs(y[eligible])).mean()) if eligible.any() else None,
            'negative_hours': int((y < 0).sum()), 'negative_mae': float(err[y < 0].mean()) if (y < 0).any() else None}


def fit_predict(x, y, data, train_days, eval_days):
    train = x.loc[train_days]
    xt = x.loc[eval_days]
    multi = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    multi.fit(train, y.loc[train_days])
    curve = pd.DataFrame(multi.predict(xt), index=eval_days, columns=range(24))
    # Hourly model has the same daily information, plus clock-hour encoding.
    def hourly(days):
        d = data[data.day.isin(days)].copy()
        xx = x.reindex(pd.DatetimeIndex(d.day)).reset_index(drop=True)
        xx['hour_sin'] = np.sin(2*np.pi*d.hour.to_numpy()/24)
        xx['hour_cos'] = np.cos(2*np.pi*d.hour.to_numpy()/24)
        return d, xx
    tr, hx = hourly(train_days)
    ev, ex = hourly(eval_days)
    boost = HistGradientBoostingRegressor(max_iter=150, learning_rate=.05, max_leaf_nodes=15,
                                            min_samples_leaf=100, l2_regularization=1, early_stopping=False, random_state=42)
    boost.fit(hx, tr.DK1)
    pred = ev[['day', 'hour', 'DK1']].copy()
    pred['previous_day'] = x.loc[pd.DatetimeIndex(ev.day)].to_numpy()[np.arange(len(ev)), ev.hour.to_numpy()]
    pred['ridge_curve'] = curve.loc[pd.DatetimeIndex(ev.day)].to_numpy()[np.arange(len(ev)), ev.hour.to_numpy()]
    pred['boosting_hourly'] = boost.predict(ex)
    pred.index = ev.index
    return pred, {'ridge_curve': multi, 'boosting_hourly': boost}


def run(forecast_path=None):
    global OUT
    if forecast_path:
        OUT = Path('results/dk1_preauction_with_forecasts')
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    x, y, data = make_daily(prices)
    if forecast_path:
        forecasts = select_vintages(pd.read_csv(forecast_path), prices.index)
        if forecasts.empty or forecasts.shape[1] == 0:
            raise ValueError('No pre-cutoff forecast vintages available')
        local = forecasts.index.tz_convert('Europe/Copenhagen')
        forecasts['day'] = local.tz_localize(None).normalize()
        forecasts['hour'] = local.hour
        fields = []
        for col in forecasts.columns.drop(['day', 'hour']):
            curve = forecasts.pivot_table(index='day', columns='hour', values=col, aggfunc='mean').reindex(index=x.index, columns=range(24))
            complete = forecasts.groupby('day')[col].apply(lambda s: s.notna().all())
            # Interpolate only the structurally absent spring clock hour, not missing records.
            for day in curve.index[curve.isna().any(axis=1)]:
                missing = curve.loc[day].isna()
                if missing.sum() == 1 and missing.iloc[2] and len(data[data.day.eq(day)]) == 23:
                    curve.loc[day, 2] = (curve.loc[day, 1] + curve.loc[day, 3]) / 2
            curve.loc[~complete.reindex(curve.index, fill_value=False)] = np.nan
            curve.columns = [f'forecast_{col}_h{h}' for h in range(24)]
            fields.append(curve)
        x = pd.concat([x] + fields, axis=1)
    valid = x.notna().all(axis=1) & y.notna().all(axis=1)
    # Drop partial days at archive edges (and fail if any internal day is missing).
    counts = data.groupby('day').size()
    expected = pd.Series([(pd.Timestamp(d + pd.Timedelta(days=1), tz='Europe/Copenhagen').tz_convert('UTC') - pd.Timestamp(d, tz='Europe/Copenhagen').tz_convert('UTC')).total_seconds()/3600 for d in counts.index], index=counts.index)
    valid &= counts.reindex(x.index).eq(expected.reindex(x.index))
    days = x.index[valid]
    splits = [('development', days[days < DEV], days[(days >= DEV) & (days < TEST)]),
              ('exploratory_test', days[days < TEST], days[(days >= TEST) & (days < END)])]
    allpred, metrics, models = [], [], {}
    for label, train, ev in splits:
        if (len(ev) != (TEST-DEV).days if label == 'development' else len(ev) != (END-TEST).days):
            raise ValueError('evaluation days incomplete')
        pred, models[label] = fit_predict(x, y, data, train, ev)
        pred['split'] = label
        for name in ['previous_day', 'ridge_curve', 'boosting_hourly']:
            for period, g in [('all', pred)] + [(str(yr), g) for yr, g in pred.groupby(pred.day.dt.year)]:
                metrics.append({'split':label, 'period':period, 'model':name, **metric(g.DK1, g[name])})
        allpred.append(pred)
        print(label, 'train days', len(train), 'eval days', len(ev), flush=True)
    metrics = pd.DataFrame(metrics)
    devmetrics = metrics[(metrics.split == 'development') & (metrics.period == 'all')]
    selected = devmetrics.sort_values(['rate_5pct', 'mae'], ascending=[False, True]).iloc[0].model
    metrics.to_csv(OUT/'metrics.csv', index=False)
    pd.concat(allpred).rename_axis('HourUTC').to_csv(OUT/'predictions.csv.gz', compression={'method':'gzip','mtime':0})
    joblib.dump({'models':models, 'columns':list(x.columns), 'selected':selected}, OUT/'models.joblib', compress=3)
    protocol = {'target':'DK1 hourly mean day-ahead EUR/MWh', 'cutoff':'10:00 Europe/Copenhagen on delivery day minus one',
                'inputs':'previous-day and previous-week full price curves for DK1, DK2, DE-LU, NL, SE4; calendar',
                'forecast_file': str(forecast_path) if forecast_path else None,
                'development':'2024-01-01 to 2025-11-09; train before 2024', 'exploratory_test':'2025-11-10 to 2026-11-09; train before test',
                'selection':'highest development ±5% hit rate for abs(actual)>=20 EUR/MWh; MAE tie-breaker', 'selected':selected,
                'caveats':['All periods previously explored; no independent confirmatory test.', 'Prior-day prices assume day-ahead results were published before next-day 10:00; historical exceptional delays not audited.', 'Current price archives may include revisions.', 'DST repeated clock hour averaged and missing clock hour interpolated for curve training; actual hours evaluated separately.', '15-minute market prices averaged hourly; not a quarter-hour forecast.', 'Timestamp filtering alone does not prove the forecast archive is immutable or that availability timestamps are accurate.', 'No weather, fuel or TSO forecasts included unless an explicit forecast file is supplied.']}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf8')
    print(metrics[metrics.period.eq('all')].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecasts', type=Path, help='CSV of timestamped, historically retained forecast vintages')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.forecasts)
