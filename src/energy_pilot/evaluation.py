"""Frozen train-only clustering and synchronous moving-day block bootstrap."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import FEATURES
from .mining import score_rule


def fit_clusters(train, test):
    scaler = StandardScaler().fit(train[FEATURES])
    x = scaler.transform(train[FEATURES])
    diagnostics, models = [], {}
    for k in range(3,8):
        model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(x)
        other = KMeans(n_clusters=k, n_init=10, random_state=73).fit(x)
        sizes = np.bincount(model.labels_, minlength=k)
        diagnostics.append(dict(k=k, silhouette=silhouette_score(x, model.labels_, sample_size=min(2000,len(x)), random_state=42),
                                davies_bouldin=davies_bouldin_score(x, model.labels_),
                                min_cluster_fraction=sizes.min()/len(x),
                                seed_stability_ari=adjusted_rand_score(model.labels_, other.labels_)))
        models[k] = model
    diagnostic = pd.DataFrame(diagnostics)
    valid = diagnostic[diagnostic.min_cluster_fraction >= .03]
    if valid.empty:
        raise ValueError('All k candidates have clusters below 3%; inspect diagnostics')
    # Deterministic tradeoff: max silhouette among non-tiny solutions, k breaks ties.
    k = int(valid.sort_values(['silhouette','k'], ascending=[False, True]).iloc[0].k)
    model = models[k]
    train, test = train.copy(), test.copy()
    train['cluster'], test['cluster'] = model.labels_, model.predict(scaler.transform(test[FEATURES]))
    centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=FEATURES)
    centers.index.name = 'cluster'
    return train, test, diagnostic, centers, scaler, model


def bootstrap_lift(frame, baskets, antecedent, event, repeats=400, block_days=7, seed=42):
    """Sample overlapping consecutive local-day blocks; keep both areas synchronous.

    Full local calendar dates, including DST days and days missing observations, are
    indexed before sampling. The base rate is recomputed in every replicate.
    """
    dates = pd.date_range(min(frame.local_date), max(frame.local_date), freq='D').date
    if len(dates) < block_days * 2:
        return dict(low=np.nan, high=np.nan, valid=0, repeats=repeats, block_days=block_days)
    index = {date: i for i,date in enumerate(dates)}
    row_days = np.array([index[d] for d in frame.local_date])
    rng, lifts = np.random.default_rng(seed), []
    for _ in range(repeats):
        starts = rng.integers(0, len(dates)-block_days+1, size=int(np.ceil(len(dates)/block_days)))
        selected = np.concatenate([np.arange(s,s+block_days) for s in starts])[:len(dates)]
        weights = np.bincount(selected, minlength=len(dates))[row_days]
        lift = score_rule(frame,baskets,antecedent,event,weights)['lift']
        if np.isfinite(lift):
            lifts.append(lift)
    low, high = np.quantile(lifts,[.025,.975]) if len(lifts) >= .8*repeats else (np.nan,np.nan)
    return dict(low=float(low), high=float(high), valid=len(lifts), repeats=repeats, block_days=block_days)


def calendar_comparison(train, test, event):
    """2023-fitted cell event rates; evaluate all test hours by Brier score.

    Unseen cells back off to the training area rate, then training pooled rate.
    Energy model adds cluster to the same calendar grouping. No test tuning.
    """
    rows = []
    base = test.PriceArea.map(train.groupby('PriceArea')[event].mean()).fillna(train[event].mean())
    for name, keys in [('calendar', ['PriceArea','season','time_of_day']),
                       ('calendar_plus_cluster', ['PriceArea','season','time_of_day','cluster'])]:
        rates = train.groupby(keys, observed=True)[event].mean()
        index = pd.MultiIndex.from_frame(test[keys])
        predicted = pd.Series(rates.reindex(index).to_numpy(), index=test.index)
        fallback_count = int(predicted.isna().sum())
        predicted = predicted.fillna(base)
        rows.append(dict(event=event, model=name, n=len(test), fallback_rows=fallback_count,
                         brier=float(np.mean((predicted-test[event].astype(float))**2))))
    return rows
