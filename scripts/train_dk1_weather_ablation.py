"""Paired DK1 price-only versus archived JMA weather experiments.

JMA previous_day2 is a fixed-lead proxy; historical publication/vintage times
are not established, so this experiment must not be labelled operational.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_dk1_preauction as base
import train_latest_forecast as archive

OUT = Path('results/dk1_weather_ablation')
VARIABLES = ['temperature_2m', 'wind_speed_10m', 'cloud_cover']


def weather_curves(prices, load_weather=archive.weather_all):
    """Create full-day weather forecasts without filling missing observations."""
    local = prices.index.tz_convert('Europe/Copenhagen')
    index_days = local.tz_localize(None).normalize()
    days = pd.DatetimeIndex(sorted(index_days.unique()))
    fields, cover = [], []
    for zone in base.ZONES:
        weather = load_weather(zone).reindex(prices.index)
        if weather.index.has_duplicates:
            raise ValueError('Duplicate weather target times')
        weather['day'], weather['hour'] = index_days, local.hour
        for variable in VARIABLES:
            values = pd.to_numeric(weather[variable], errors='raise')
            complete = values.notna().groupby(weather.day).all().reindex(days, fill_value=False)
            w = weather[['day', 'hour', variable]]
            curve = w.pivot_table(index='day', columns='hour', values=variable, aggfunc='mean').reindex(index=days, columns=range(24))
            counts = weather.groupby('day').size().reindex(days, fill_value=0)
            for day in curve.index[(counts == 23) & complete]:
                if curve.loc[day].isna().sum() == 1 and pd.isna(curve.loc[day, 2]):
                    curve.loc[day, 2] = (curve.loc[day, 1] + curve.loc[day, 3]) / 2
            curve.loc[~complete] = np.nan
            curve.columns = [f'weather_{zone}_{variable}_h{h}' for h in range(24)]
            fields.append(curve)
            cover.append({'zone': zone, 'variable': variable, 'hours':len(values), 'missing_hours':int(values.isna().sum()),
                          'complete_days':int(complete.sum())})
    return pd.concat(fields, axis=1), pd.DataFrame(cover)


def paired_intervals(pred, left, right, seed=42, repeats=2000, block=7):
    """Paired moving-week bootstrap; intervals conditional on fitted models."""
    y = pred.DK1.to_numpy()
    eligible = np.abs(y) >= 20
    a, b = pred[left].to_numpy(), pred[right].to_numpy()
    day = pd.DataFrame({'day': pred.day.to_numpy(), 'n': 1, 'eligible': eligible.astype(int),
        'mae_improvement':np.abs(a-y)-np.abs(b-y),
        'hit_improvement':((np.abs(b-y)<=.05*np.abs(y)) & eligible).astype(int) -
                          ((np.abs(a-y)<=.05*np.abs(y)) & eligible).astype(int)})
    daily = day.groupby('day').sum().to_numpy()
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        starts = rng.integers(0, len(daily)-block+1, size=int(np.ceil(len(daily)/block)))
        idx = np.concatenate([np.arange(s, s+block) for s in starts])[:len(daily)]
        n, eligible_n, improvement, hits = daily[idx].sum(axis=0)
        values.append([improvement/n, hits/eligible_n])
    return np.quantile(values, [.025, .975], axis=0)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    prices = base.load_prices()
    x, y, data = base.make_daily(prices)
    weather, coverage = weather_curves(prices)
    coverage.to_csv(OUT/'weather_coverage.csv', index=False)
    with_weather = pd.concat([x, weather], axis=1)
    counts = data.groupby('day').size().reindex(x.index)
    expected = pd.Series([(pd.Timestamp(d + pd.Timedelta(days=1), tz='Europe/Copenhagen').tz_convert('UTC') -
                           pd.Timestamp(d, tz='Europe/Copenhagen').tz_convert('UTC')).total_seconds()/3600
                          for d in x.index], index=x.index)
    valid = x.notna().all(axis=1) & with_weather.notna().all(axis=1) & y.notna().all(axis=1) & counts.eq(expected)
    days = x.index[valid]
    splits = [('development', days[days < base.DEV], days[(days >= base.DEV) & (days < base.TEST)]),
              ('exploratory_test', days[days < base.TEST], days[(days >= base.TEST) & (days < base.END)])]
    metrics, predictions, models, intervals = [], [], {}, []
    for label, train, evaluation in splits:
        if len(evaluation) != ((base.TEST-base.DEV).days if label=='development' else (base.END-base.TEST).days):
            raise ValueError('Incomplete common evaluation days')
        variants = {}
        for group, features in [('price', x), ('price_weather', with_weather)]:
            print('fit', label, group, len(train), 'train days', flush=True)
            pred, fitted = base.fit_predict(features, y, data, train, evaluation)
            variants[group] = pred
            models[f'{label}_{group}'] = {'models':fitted, 'columns':list(features.columns)}
        a, b = variants['price'], variants['price_weather']
        if not a.index.equals(b.index) or not a.DK1.equals(b.DK1):
            raise ValueError('Price and weather evaluations are not paired')
        p = a[['day','hour','DK1','previous_day']].copy()
        for group, variant in variants.items():
            for family in ['ridge_curve','boosting_hourly']:
                p[f'{family}_{group}'] = variant[family]
        p['split'] = label
        predictions.append(p)
        for model in p.columns.drop(['day','hour','DK1','split']):
            for period, g in [('all', p)] + [(str(year),g) for year,g in p.groupby(p.day.dt.year)]:
                metrics.append({'split':label, 'period':period, 'model':model, **base.metric(g.DK1,g[model])})
        for family in ['ridge_curve','boosting_hourly']:
            ci = paired_intervals(p, family+'_price', family+'_price_weather')
            intervals.append({'split':label,'family':family,'mae_improvement_low':ci[0,0], 'mae_improvement_high':ci[1,0],
                              'hit_rate_improvement_low':ci[0,1], 'hit_rate_improvement_high':ci[1,1],
                              'block_days':7, 'repeats':2000})
    metrics = pd.DataFrame(metrics)
    dev = metrics[(metrics.split == 'development') & (metrics.period == 'all')]
    selected = dev.sort_values(['rate_5pct','mae'], ascending=[False,True]).iloc[0].model
    metrics.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    pd.concat(predictions).rename_axis('HourUTC').to_csv(OUT/'predictions.csv.gz', compression={'method':'gzip','mtime':0})
    joblib.dump({'fits':models, 'selected':selected},OUT/'models.joblib',compress=3)
    protocol = {'target':'DK1 hourly averaged day-ahead price EUR/MWh', 'forecast_cutoff':'D-1 10:00 Europe/Copenhagen',
      'weather':'Open-Meteo JMA GSM previous_day2 48h lead at one city per zone; temperature, wind speed and cloud cover',
      'warning':'Individual publication times and historical revisions not verified. Nominal lead is not proof of availability.',
      'train_development':'2019-01-08 to 2023-12-31 on common complete days',
      'development':'2024-01-01 to 2025-11-09', 'train_test':'all common complete days before 2025-11-10',
      'test':'2025-11-10 to 2026-11-09; already explored, not independent',
      'train_days_by_split':{label:len(train) for label,train,_ in splits},
      'evaluation_days_by_split':{label:len(ev) for label,_,ev in splits},
      'selection':'highest development +/-5% rate for abs(actual)>=20, MAE tie-breaker', 'selected':selected,
      'comparison':'same training dates, test hours, model configuration and price features within each family',
      'caveats':['No validated TSO pre-auction forecast vintages included.', 'Prior-day price publication assumed; revisions not audited.',
                 '15-minute auction prices aggregated to hourly average.', 'DST repeated hour averaged and spring missing hour interpolated for curves.',
                 'Bootstrap conditional on fitted models; no adjustment for selection or past explorations.']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
    print(metrics[metrics.period.eq('all')].to_string(index=False), flush=True)
    print(pd.DataFrame(intervals).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
