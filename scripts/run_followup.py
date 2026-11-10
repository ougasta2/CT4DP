"""Exploratory expanding-window validation using the bundled decade table."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from energy_pilot.data import Features
from energy_pilot.evaluation import fit_clusters
from energy_pilot.pipeline import select_rules
from energy_pilot.mining import baskets, score_rule
from energy_pilot.decade_evaluation import cell_predictions, fast_block_interval


def matched_cells(test, solar_cluster, year):
    """Compare clusters within area, season, and time-of-day, without pooling cells."""
    rows = []
    for (area, season, tod), g in test.groupby(['PriceArea', 'season', 'time_of_day'], observed=True):
        high = g.cluster.eq(solar_cluster)
        a, b = g[high], g[~high]
        na, nb = len(a), len(b)
        ea, eb = int(a.negative_price.sum()), int(b.negative_price.sum())
        rows.append(dict(year=year, area=area, season=season, time_of_day=tod,
                         solar_cluster=solar_cluster, solar_n=na, solar_events=ea,
                         other_n=nb, other_events=eb,
                         solar_rate=ea/na if na else np.nan,
                         other_rate=eb/nb if nb else np.nan,
                         rate_difference=ea/na-eb/nb if na and nb else np.nan))
    return rows


def run():
    out = Path('results/followup'); out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv('results/decade/processed.csv.gz', parse_dates=['HourUTC'])
    frame['local_date'] = pd.to_datetime(frame.local_date).dt.date
    rules, cells, metrics, centers_all, diagnostics, thresholds = [], [], [], [], [], []
    for year in range(2020, 2026):
        raw_train = frame[frame.year < year].copy()
        raw_test = frame[frame.year == year].copy()
        print(f'Expanding window: 2016-{year-1} -> {year}', flush=True)
        transform = Features().fit(raw_train)
        train = transform.transform(raw_train, solar_total=True)
        test = transform.transform(raw_test, solar_total=True)
        train, test, diag, centers, _, model = fit_clusters(train, test)
        solar_cluster = int(centers.solar_to_load.idxmax())
        diag['test_year'] = year; diagnostics.append(diag)
        centers = centers.reset_index(); centers['test_year'] = year
        centers['highest_solar'] = centers.cluster.eq(solar_cluster); centers_all.append(centers)
        chosen, _, _ = select_rules(train, .02)
        train_b, test_b = baskets(train), baskets(test)
        for event, antecedent in chosen:
            if event != 'negative_price':
                continue
            a = score_rule(train, train_b, antecedent, event)
            b = score_rule(test, test_b, antecedent, event)
            row = dict(test_year=year, train_start=2016, train_end=year-1,
                       k=model.n_clusters, antecedent=' & '.join(sorted(antecedent)),
                       **{'train_'+k:v for k,v in a.items()}, **{'test_'+k:v for k,v in b.items()})
            for days in (3, 7, 14):
                interval = fast_block_interval(test, test_b, antecedent, event, repeats=400, block_days=days)
                row.update({f'ci_{days}d_low':interval['low'], f'ci_{days}d_high':interval['high'],
                            f'ci_{days}d_valid':interval['valid']})
            rules.append(row)
        cells.extend(matched_cells(test, solar_cluster, year))
        for area, g in test.groupby('PriceArea'):
            for cluster in (False, True):
                p, fallback = cell_predictions(train, g, 'negative_price', cluster=cluster)
                metrics.append(dict(test_year=year, area=area, model='calendar+cluster' if cluster else 'calendar',
                                    n=len(g), events=int(g.negative_price.sum()), fallback=fallback,
                                    brier=float(np.mean((p-g.negative_price.to_numpy())**2))))
            recent = raw_train[(raw_train.year >= year-3) & (raw_train.PriceArea == area)]
            q = recent.SpotPriceEUR.quantile(.95)
            fixed = transform.thresholds[area]
            thresholds.append(dict(test_year=year, area=area, recent_train_start=year-3,
                                   recent_train_end=year-1, recent_q95=q, expanding_q95=fixed,
                                   n=len(g), recent_events=int((g.SpotPriceEUR>q).sum()),
                                   expanding_events=int((g.SpotPriceEUR>fixed).sum())))
    pd.DataFrame(rules).to_csv(out/'rolling_rules.csv', index=False)
    pd.DataFrame(cells).to_csv(out/'matched_calendar_cells.csv', index=False)
    pd.DataFrame(metrics).to_csv(out/'rolling_brier.csv', index=False)
    pd.concat(centers_all).to_csv(out/'rolling_centers.csv', index=False)
    pd.concat(diagnostics).to_csv(out/'rolling_k_diagnostics.csv', index=False)
    pd.DataFrame(thresholds).to_csv(out/'high_price_threshold_sensitivity.csv', index=False)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
