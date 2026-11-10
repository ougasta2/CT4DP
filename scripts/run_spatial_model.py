"""Exploratory spatial discordance model; not a pre-auction forecast."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from energy_pilot.mining import apriori, mask_for

FEATURES = ['wind_to_load', 'solar_to_load', 'load_index']
COMMON = [f'common_{x}' for x in FEATURES]
DIFF = [f'diff_{x}' for x in FEATURES]
CAL = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'weekend']


def paired(frame):
    if frame.duplicated(['HourUTC', 'PriceArea']).any():
        raise ValueError('Duplicate area-hour keys')
    cols = FEATURES + ['SpotPriceEUR']
    wide = frame.pivot(index='HourUTC', columns='PriceArea', values=cols)
    if set(wide.columns.get_level_values(1)) != {'DK1', 'DK2'} or wide.isna().any().any():
        raise ValueError('Every hour requires complete DK1 and DK2 observations')
    out = pd.DataFrame(index=wide.index)
    for f in FEATURES:
        out['common_'+f] = (wide[f]['DK1'] + wide[f]['DK2']) / 2
        out['diff_'+f] = wide[f]['DK1'] - wide[f]['DK2']
    for area in ['DK1', 'DK2']:
        out['price_'+area] = wide['SpotPriceEUR'][area]
    out.index = pd.to_datetime(out.index, utc=True)
    local = out.index.tz_convert('Europe/Copenhagen')
    out['local_date'] = local.date
    out['year'] = local.year
    out['hour_sin'] = np.sin(2*np.pi*local.hour/24)
    out['hour_cos'] = np.cos(2*np.pi*local.hour/24)
    out['month_sin'] = np.sin(2*np.pi*(local.month-1)/12)
    out['month_cos'] = np.cos(2*np.pi*(local.month-1)/12)
    out['weekend'] = (local.weekday >= 5).astype(int)
    out['season'] = np.select([local.month.isin([12,1,2]), local.month.isin([3,4,5]),
                               local.month.isin([6,7,8])], ['winter','spring','summer'], default='autumn')
    out['time'] = pd.cut(local.hour, [-1,5,11,17,23], labels=['night','morning','afternoon','evening']).astype(str)
    a, b = out.price_DK1 < 0, out.price_DK2 < 0
    out['any_negative'] = a | b
    out['discordant'] = a ^ b
    out['state'] = np.select([a & b, a & ~b, ~a & b], ['both','DK1_only','DK2_only'], default='neither')
    return out


def block_delta(frame, differences, repeats=1000, days=7):
    # The date grid covers the whole year, including dates with no negative hour.
    year = int(frame.year.iloc[0])
    grid = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D').date
    daily = pd.DataFrame({'date':frame.local_date, 'delta':differences, 'n':1}).groupby('date').sum().reindex(grid, fill_value=0).to_numpy()
    rng = np.random.default_rng(42); values = []
    for _ in range(repeats):
        starts = rng.integers(0, len(grid)-days+1, int(np.ceil(len(grid)/days)))
        idx = np.concatenate([np.arange(s, s+days) for s in starts])[:len(grid)]
        delta, n = daily[idx].sum(axis=0)
        if n: values.append(delta/n)
    return np.quantile(values, [.025,.975]) if len(values) >= .8*repeats else (np.nan,np.nan)


def transactions(frame):
    return [frozenset([f'cluster=C{c}', f'season={s}', f'time={t}'])
            for c, s, t in zip(frame.cluster, frame.season, frame.time)]


def run():
    out = Path('results/spatial'); out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv('results/decade/processed.csv.gz')
    data = paired(raw)
    train_all = data[data.year.between(2016, 2023)].copy()
    scaler = StandardScaler().fit(train_all[COMMON+DIFF])
    model = KMeans(n_clusters=4, n_init=20, random_state=42).fit(scaler.transform(train_all[COMMON+DIFF]))
    data['cluster'] = model.predict(scaler.transform(data[COMMON+DIFF]))
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=COMMON+DIFF)
    centers.index.name = 'cluster'; centers.to_csv(out/'cluster_centers.csv')
    data.groupby(['year','state']).size().unstack(fill_value=0).to_csv(out/'annual_states.csv')
    train = data[data.year.between(2016,2023) & data.any_negative].copy()
    y = train.discordant.astype(int).to_numpy()
    inputs = {'calendar':CAL, 'common':CAL+COMMON, 'common_and_difference':CAL+COMMON+DIFF}
    predictions = {}; fitted = {}
    for name, features in inputs.items():
        z = StandardScaler().fit(train[features])
        clf = LogisticRegression(C=1., max_iter=2000, random_state=42).fit(z.transform(train[features]), y)
        fitted[name] = (z, clf, features)
    baskets = transactions(train)
    candidates = []
    for ant, count in apriori(baskets, min_support=.05, max_size=3).items():
        if not any(x.startswith('cluster=') for x in ant): continue
        mask = mask_for(baskets, ant); events = int(y[mask].sum())
        if count < 40 or events < 20: continue
        candidates.append((events/count / y.mean(), count, tuple(sorted(ant))))
    chosen = max(candidates, default=None)
    metrics, intervals, rule_rows = [], [], []
    for year in [2024,2025]:
        test = data[(data.year == year) & data.any_negative].copy()
        target = test.discordant.astype(int).to_numpy()
        p = {'training_rate': np.full(len(test), y.mean())}
        for name,(z,clf,features) in fitted.items():
            p[name] = clf.predict_proba(z.transform(test[features]))[:,1]
        for subset, keep in [('full_year', np.ones(len(test), dtype=bool)),
                             ('jan_sep', np.asarray(test.index.month <= 9))]:
            if year != 2025 and subset == 'jan_sep': continue
            for name, values in p.items():
                actual, prob = target[keep], values[keep]
                metrics.append(dict(year=year, period=subset, model=name,n=len(actual),events=int(actual.sum()),
                                    prevalence=actual.mean(),brier=brier_score_loss(actual,prob),
                                    log_loss=log_loss(actual,prob),auc=roc_auc_score(actual,prob),
                                    average_precision=average_precision_score(actual,prob)))
        for baseline in ['training_rate', 'calendar', 'common']:
            delta = (p[baseline]-target)**2-(p['common_and_difference']-target)**2
            lo,hi = block_delta(test,delta)
            intervals.append(dict(year=year,comparison=baseline+' minus common_and_difference Brier',
                                  improvement=float(delta.mean()),ci_low=lo,ci_high=hi,block_days=7,repeats=1000))
        if chosen:
            antecedent = frozenset(chosen[2]); mask = mask_for(transactions(test), antecedent)
            n = int(mask.sum()); events = int(target[mask].sum())
            rule_rows.append(dict(year=year,antecedent=' & '.join(chosen[2]),n=n,events=events,
                                  rate=events/n if n else np.nan,base_rate=target.mean(),
                                  lift=events/n/target.mean() if n and target.mean() else np.nan))
        predictions[year] = test.assign(**{f'p_{k}':v for k,v in p.items()})[['local_date','state','discordant','cluster']+[f'p_{k}' for k in p]]
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_brier_intervals.csv',index=False)
    pd.DataFrame(rule_rows).to_csv(out/'selected_rule.csv',index=False)
    pd.concat(predictions.values()).to_csv(out/'predictions.csv.gz',compression={'method':'gzip','mtime':0})
    (out/'protocol.json').write_text(json.dumps(dict(train='2016-2023',test=[2024,2025],
        target='exactly one negative conditional on at least one negative',
        train_n=len(train),train_events=int(y.sum()), cluster_fit='all training hours, no prices',
        k=4, logistic_C=1, selected_rule=chosen, note='exploratory; test years previously inspected'),indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(limits=2): run()
