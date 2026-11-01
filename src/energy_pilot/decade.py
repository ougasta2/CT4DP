"""2016–2025 pilot: eight training years, two test years, four additional methods."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .api import Client, KEYS, COLUMNS, PRODUCTION
from .data import Features,FEATURES
from .decade_data import load_decade,chronological_split,TRAIN_YEARS,TEST_YEARS
from .evaluation import fit_clusters
from .decade_methods import (clustering_extensions,cluster_diagnostics,fit_supervised,feature_matrix,
                             RF_CONFIG,REG_CONFIG)
from .decade_evaluation import (classification_metrics,regression_metrics,cell_predictions,
                                evaluate_rules,paired_error_interval)
from .decade_plots import plot_decade
from .pipeline import save_json,markdown_table,select_rules


def write_findings(out,events,classification,regression,rules,improvements,cluster_stats,usable):
    pooled=classification[(classification.test_year=='2024-2025')&(classification.area=='both')]
    reg=regression[(regression.test_year=='2024-2025')&(regression.area=='both')]
    rule_table=rules.copy()
    if len(rule_table):
        rule_table['test_events/hours']=[f'{int(e)}/{int(n)}' for e,n in zip(rule_table.test_events,rule_table.test_n)]
    rf=pooled[pooled.method=='random_forest'].set_index(['event','features'])
    neg_base=rf.loc[('negative_price','calendar_only'),'brier']
    neg_energy=rf.loc[('negative_price','energy_calendar'),'brier']
    high_base=rf.loc[('high_price','calendar_only'),'brier']
    high_energy=rf.loc[('high_price','energy_calendar'),'brier']
    gb=reg[reg.method=='gradient_boosting'].set_index('features')
    training_events=events[events.year.isin(TRAIN_YEARS)].high_events.sum()
    share2022=events[events.year==2022].high_events.sum()/training_events
    k_neg=rules[(rules.method=='kmeans')&(rules.event=='negative_price')]
    lines=['# Ten-year pilot findings','',
           '**Training: 2016–2023 (eight years). Test: 2024–2025 (two later years).**',
           'The latest ten completed calendar years as of November 1, 2026 are 2016–2025. Source coverage is checked before fitting; no random train/test split is used.',
           '',f'Usable training area-hours: {int(usable[usable.year.isin(TRAIN_YEARS)].usable.sum()):,}; test area-hours: {int(usable[usable.year.isin(TEST_YEARS)].usable.sum()):,}.',
           '', '## Main reading', '',
           f'Random Forest negative-event Brier improves from {neg_base:.6f} (calendar only) to {neg_energy:.6f} (energy + calendar), a {(neg_base-neg_energy)/neg_base:.1%} relative reduction. High-event Brier worsens from {high_base:.6f} to {high_energy:.6f}. The paired intervals below quantify those changes.',
           f'Gradient-boosting MAE changes from {gb.loc["calendar_only","mae_eur_mwh"]:.2f} to {gb.loc["energy_calendar","mae_eur_mwh"]:.2f} EUR/MWh when energy features are added. More realized-energy inputs do not uniformly improve out-of-time performance.',
           'The fixed K-means afternoon/solar-rich negative-price rules have pooled test lifts '+', '.join(f'{row.antecedent.split(" & ")[0]}: {row.test_lift:.2f} ({int(row.test_events)}/{int(row.test_n)} events/hours)' for row in k_neg.itertuples())+'. See rule_by_test_year.csv for each year separately.',
           f'{share2022:.1%} of training high-price events occur in 2022. Their pooled test prevalence is only {pooled[pooled.event=="high_price"].base_rate.iloc[0]:.2%}, versus about 5% in pooled training by definition. This documents a substantial target-distribution shift; it does not identify a causal explanation.',
           'All methods use realized settlement generation/load. Classifiers and regressors are retrospective conditional models, **not operational day-ahead forecasts**. The 2024 period was already examined in the original pilot, so this is exploratory temporal evaluation, not an untouched confirmatory holdout.',
           '', '## Data compatibility', '',
           'Before 2020, the source places all solar in SolarPowerSelfConMWh. From 2020, total reported solar is grid components plus estimated self-consumption. The decade uses this **reported total solar proxy**, not a fictitious pre-2020 grid-solar series. Metadata warns of a measurement/layout break and uncertain gross-consumption accounting. See the separate wind/load-only sensitivity and annual profiles.',
           'The 2020 data contain 1,747 negative estimated self-use values, despite the metadata definition max(modeled production minus actual, 0). The harmonized feature applies that stated nonnegative bound, preserves the original signed values and flags all repairs. This is a documented repair assumption, not zero-filling missing observations. A separate strict clustering sensitivity excludes every repaired row and still represents all eight training years; event definitions and the test population stay fixed.',
           'From October 1, 2025, four complete 15-minute prices are averaged to EUR/MWh per hour. Negative/high events then concern the hourly mean; a negative quarter need not make its hourly mean negative. Exact quarter counts, original responses and the transition audit are retained. Prices are not summed.',
           '', '## Event counts and fixed thresholds', '',markdown_table(events),
           '', 'High-price thresholds are the area-specific pooled 2016–2023 95th percentiles, held fixed for both test years. This differs from the original 2023-only threshold.',
           '', '## Unsupervised methods', '',markdown_table(cluster_stats),
           'K-means uses k=3..7, training-only standardization, silhouette and a 3% smallest-cluster guard. GMM selects among 3..7 full-covariance mixtures by training BIC, convergence and the same size guard. DBSCAN fits a deterministic <=20,000-row year/area-stratified training sample; every training/test row is assigned via frozen core neighborhoods. This is a bounded pilot approximation, not a native DBSCAN predict operation. Noise remains -1.',
           '', '## Classification: temporal holdout', '',
           markdown_table(pooled[['method','features','event','n','events','brier','average_precision','roc_auc']]),
           'Lower Brier is better. Average precision should be read against the event prevalence; ROC-AUC alone can hide rare-event weakness. Full per-year/area results and fixed-0.5 precision/recall are in classification_metrics.csv. Cluster probability cells and all calendar backoffs use training outcomes only.',
           '', '## Price regression: temporal holdout', '',
           markdown_table(reg[['method','features','n','mae_eur_mwh','rmse_eur_mwh','mean_error_eur_mwh']]),
           'MAE/RMSE are EUR/MWh; smaller is better. HistGradientBoostingRegressor uses fixed hyperparameters and early_stopping=False, avoiding random internal validation. No holdout target is used in fitting or selection.',
           '', '## Paired energy-versus-calendar comparisons', '',markdown_table(improvements),
           'Positive improvement means adding realized energy features lowers loss against the same model family using calendar features alone. Intervals use paired seven-day moving blocks of the test period, preserving both areas and serial dependence. These exploratory intervals do not account for trying multiple model families.',
           '', '## Solar-definition and correction sensitivity', '',
           markdown_table(pd.read_csv(out/'solar_break_sensitivity.csv').query("test_year == '2024-2025' and area == 'both'")[[
               'method','event','n','events','brier','average_precision']]),
           'The no-solar variant removes the measurement bridge entirely. The strict-no-repair variant retains only original nonnegative estimates in training, using the fixed primary event definitions and the same two test years. Differences are reported, not used to retune the primary models.',
           '', '## Apriori rules for the three clustering methods', '',
           markdown_table(rule_table[['method','event','antecedent','train_lift','test_events/hours','test_lift','ci_low','ci_high','test_small_sample']]) if len(rule_table) else 'No rule met the frozen training support requirements.',
           'Predictor-only Apriori retains rare outcomes. At most one training-selected rule per area/event/method; >=2% antecedent support, >=100 hours and >=20 events. Rules are frozen before holdout scoring. Year-specific counts/lifts are in rule_by_test_year.csv; do not treat pooled two-year enrichment as stability in each year.',
           'A rule with zero observed test events can have a degenerate [0,0] empirical bootstrap interval. This does not show that its future event probability is zero; small-sample flags and exact counts take priority over an apparently narrow interval.',
           '', '## Limits and next iteration', '',
           'Ten years span major market and measurement changes. Pooled thresholds, capacity growth and distribution drift can change event prevalence; no installed-capacity normalization, weather, fuel or interconnector features are added. No causal claim is supported. Cluster IDs are method-specific. DBSCAN separation failure or high noise is a result, not a reason to tune against the test set.',
           '1. Inspect 2020 solar-definition and 2025 interval-transition sensitivity before substantive interpretation.',
           '2. Check year/area scores and paired intervals rather than selecting the best pooled test score for retuning.',
           '3. If building a forecasting study later, obtain genuinely pre-clearing predictors and reserve a new untouched period.',
           '', 'Source: Energinet Energi Data Service, CC BY 4.0. Official metadata, raw selected-field responses and derived data are bundled with the study.']
    (out/'pilot_findings.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    frame,audit,merge,usable=load_decade(Client(args.cache,args.offline))
    audit.to_csv(out/'coverage.csv',index=False);usable.to_csv(out/'usable_coverage.csv',index=False)
    save_json(out/'merge_audit.json',merge)
    # Preserve all selected raw inputs and bridge provenance for GitHub/ZIP users.
    raw=Path('data/decade');raw.mkdir(parents=True,exist_ok=True)
    production_columns=COLUMNS[PRODUCTION]
    frame[production_columns].to_csv(raw/'production_2016_2025.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    frame[KEYS+['SpotPriceEUR','price_resolution','quarter_negative_fraction']].to_csv(raw/'prices_hourly_2016_2025.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    quarter=frame[frame.price_resolution=='mean_of_4_quarters']
    save_json(out/'price_transition.json',dict(hourly_rows=len(quarter),
              hourly_mean_negative=int((quarter.SpotPriceEUR<0).sum()),
              any_quarter_negative=int((quarter.quarter_negative_fraction>0).sum()),
              any_quarter_but_mean_not_negative=int(((quarter.quarter_negative_fraction>0)&(quarter.SpotPriceEUR>=0)).sum())))
    profiles=frame.groupby(['year','PriceArea']).agg(n=('SpotPriceEUR','size'),
        solar_proxy_median=('solar_harmonized_total_mwh','median'),solar_proxy_max=('solar_harmonized_total_mwh','max'),
        solar_estimate_repairs=('solar_estimate_floor_applied','sum'),solar_source_total_min=('solar_reported_total_mwh','min'),
        wind_median=('wind_mwh','median'),load_median=('gross_load_mwh','median'),
        price_min=('SpotPriceEUR','min'),price_max=('SpotPriceEUR','max')).reset_index()
    profiles.to_csv(out/'annual_profiles.csv',index=False)
    train_raw,test_raw=chronological_split(frame)
    transform=Features().fit(train_raw)
    train=transform.transform(train_raw,solar_total=True);test=transform.transform(test_raw,solar_total=True)
    print(f'TRAIN 2016-2023: {len(train)} rows; TEST 2024-2025: {len(test)} rows',flush=True)
    print('Fitting K-means baseline...',flush=True)
    train,test,k_diagnostics,k_centers,scaler,kmeans=fit_clusters(train,test)
    k_diagnostics.to_csv(out/'kmeans_candidates.csv',index=False)
    print('Fitting GMM and sampled DBSCAN...',flush=True)
    extensions,method_diagnostics,sample_index=clustering_extensions(train,test,scaler)
    method_diagnostics.to_csv(out/'additional_clustering_candidates.csv',index=False)
    pd.Series(train.iloc[sample_index].groupby(['year','PriceArea']).size(),name='sample_n').to_csv(out/'dbscan_sample_counts.csv')
    cluster_models={'kmeans':(train.cluster.to_numpy(),test.cluster.to_numpy(),kmeans),**extensions}
    labels_train={};labels_test={};selected={};candidates=[];center_rows=[];sizes=[];cluster_stats=[]
    for name,(a,b,model) in cluster_models.items():
        ta=train.copy();te=test.copy();ta['cluster']=a;te['cluster']=b
        labels_train[name]=ta;labels_test[name]=te
        selected[name],scores,itemset_count=select_rules(ta,args.min_support)
        scores['method']=name;candidates.append(scores)
        for cluster,group in ta.groupby('cluster'):
            center_rows.append(dict(method=name,cluster=int(cluster),n=len(group),**group[FEATURES].mean().to_dict()))
        counts=pd.concat([ta,te]).groupby(['year','cluster']).size().rename('n').reset_index()
        counts['fraction']=counts.n/counts.groupby('year').n.transform('sum');counts['method']=name;sizes.append(counts)
        for split,data in [('train',ta),('test',te)]:
            cluster_stats.append(dict(method=name,split=split,n=len(data),
                **cluster_diagnostics(scaler.transform(data[FEATURES]),data.cluster)))
    print('Fitting Random Forest classification and gradient boosting regression...',flush=True)
    supervised=fit_supervised(train)
    protocol=dict(training_years=TRAIN_YEARS,test_years=TEST_YEARS,
                  frozen_before_scoring_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                  medians=transform.medians,demand_quantiles={k:v.tolist() for k,v in transform.quantiles.items()},
                  high_price_thresholds_eur_mwh=transform.thresholds,features=FEATURES,
                  solar_feature='harmonized total solar proxy; pre-2020 layout bridge; negative estimated self-use floored per metadata',
                  training_solar_repairs=int(train.solar_estimate_floor_applied.sum()),
                  price_feature='hourly EUR/MWh; mean of four quarters after 2025-10-01',
                  scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
                  kmeans_k=int(kmeans.n_clusters),gmm_components=int(extensions['gmm'][2].n_components),
                  dbscan_eps=float(extensions['dbscan'][2].eps),dbscan_sample_rows=len(sample_index),
                  rf_config=RF_CONFIG,regression_config=REG_CONFIG,random_seed=42,min_support=args.min_support,
                  rules={m:[dict(event=e,antecedent=sorted(a)) for e,a in s] for m,s in selected.items()},
                  prior_holdout_exposure='2024 was examined in the original pilot; exploratory temporal evaluation')
    save_json(out/'frozen_protocol.json',protocol)
    joblib.dump(dict(transform=transform,scaler=scaler,clusters={k:v[2] for k,v in cluster_models.items()},
                     supervised=supervised),out/'fitted_models.joblib',compress=3)
    print('All primary methods and rules frozen; scoring the two held-out years...',flush=True)
    class_rows=[];reg_rows=[];rule_rows=[];rule_year_rows=[];improvements=[]
    predictions=test[KEYS+['year','SpotPriceEUR','negative_price','high_price']].copy()
    for event in ['negative_price','high_price']:
        p,fallback=cell_predictions(train,test,event)
        class_rows.extend(classification_metrics(test,event,p,'calendar_cells','calendar_only'))
        for method in cluster_models:
            p,_=cell_predictions(labels_train[method],labels_test[method],event,cluster=True)
            class_rows.extend(classification_metrics(test,event,p,method,'calendar+cluster'))
            predictions[f'{method}_{event}']=p
    for method in cluster_models:
        r,y=evaluate_rules(labels_train[method],labels_test[method],selected[method],method,args.bootstrap)
        rule_rows.extend(r);rule_year_rows.extend(y)
        predictions[method+'_cluster']=labels_test[method].cluster.to_numpy()
    for event in ['negative_price','high_price']:
        p={}
        for features,energy in [('calendar_only',False),('energy_calendar',True)]:
            model=supervised[('random_forest',features,event)]
            index=list(model.classes_).index(True)
            p[features]=model.predict_proba(feature_matrix(test,energy))[:,index]
            class_rows.extend(classification_metrics(test,event,p[features],'random_forest',features))
            predictions[f'random_forest_{features}_{event}']=p[features]
        target=test[event].to_numpy(dtype=float)
        improvements.append(dict(method='random_forest',target=event,loss='Brier',
             **paired_error_interval(test,(p['calendar_only']-target)**2,(p['energy_calendar']-target)**2,args.bootstrap)))
    baseline,_=cell_predictions(train,test,'SpotPriceEUR',median=True)
    reg_rows.extend(regression_metrics(test,baseline,'calendar_cells','calendar_only'))
    p={}
    for features,energy in [('calendar_only',False),('energy_calendar',True)]:
        p[features]=supervised[('gradient_boosting',features,'SpotPriceEUR')].predict(feature_matrix(test,energy))
        reg_rows.extend(regression_metrics(test,p[features],'gradient_boosting',features))
        predictions['regression_'+features]=p[features]
    target=test.SpotPriceEUR.to_numpy()
    improvements.append(dict(method='gradient_boosting',target='SpotPriceEUR',loss='MAE EUR/MWh',
         **paired_error_interval(test,abs(p['calendar_only']-target),abs(p['energy_calendar']-target),args.bootstrap)))
    classification=pd.DataFrame(class_rows);regression=pd.DataFrame(reg_rows);rules=pd.DataFrame(rule_rows)
    improvements=pd.DataFrame(improvements);centers=pd.DataFrame(center_rows);sizes=pd.concat(sizes)
    cluster_stats=pd.DataFrame(cluster_stats)
    pd.concat(candidates).to_csv(out/'candidate_rules_train.csv',index=False)
    classification.to_csv(out/'classification_metrics.csv',index=False)
    regression.to_csv(out/'regression_metrics.csv',index=False)
    rules.to_csv(out/'selected_rules.csv',index=False);pd.DataFrame(rule_year_rows).to_csv(out/'rule_by_test_year.csv',index=False)
    improvements.to_csv(out/'paired_improvement_intervals.csv',index=False)
    centers.to_csv(out/'cluster_profiles.csv',index=False);sizes.to_csv(out/'cluster_year_sizes.csv',index=False)
    cluster_stats.to_csv(out/'cluster_metrics.csv',index=False)
    predictions.to_csv(out/'holdout_predictions.csv.gz',index=False)
    events=pd.concat([train,test]).groupby(['year','PriceArea']).agg(n=('SpotPriceEUR','size'),
                negative_events=('negative_price','sum'),high_events=('high_price','sum')).reset_index()
    events['negative_rate']=events.negative_events/events.n;events['high_rate']=events.high_events/events.n
    events.to_csv(out/'event_rates.csv',index=False)
    pd.concat([train,test]).to_csv(out/'processed.csv.gz',index=False)
    # Measurement-break sensitivity: remove solar and refit on the SAME 8 training years.
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    wind_features=['wind_to_load','load_index']
    wind_scaler=StandardScaler().fit(train[wind_features])
    wind_model=KMeans(n_clusters=kmeans.n_clusters,n_init=20,random_state=42).fit(wind_scaler.transform(train[wind_features]))
    wind_train=train.copy();wind_test=test.copy()
    wind_train['cluster']=wind_model.labels_;wind_test['cluster']=wind_model.predict(wind_scaler.transform(test[wind_features]))
    sensitivity=[]
    for event in ['negative_price','high_price']:
        probabilities,_=cell_predictions(wind_train,wind_test,event,cluster=True)
        sensitivity.extend(classification_metrics(wind_test,event,probabilities,'kmeans_without_solar','calendar+cluster'))
    strict_train=train[~train.solar_estimate_floor_applied].copy()
    strict_scaler=StandardScaler().fit(strict_train[FEATURES])
    strict_model=KMeans(n_clusters=kmeans.n_clusters,n_init=20,random_state=42).fit(strict_scaler.transform(strict_train[FEATURES]))
    strict_train['cluster']=strict_model.labels_
    strict_test=test.copy();strict_test['cluster']=strict_model.predict(strict_scaler.transform(test[FEATURES]))
    for event in ['negative_price','high_price']:
        probabilities,_=cell_predictions(strict_train,strict_test,event,cluster=True)
        sensitivity.extend(classification_metrics(strict_test,event,probabilities,'kmeans_strict_no_solar_repair','calendar+cluster'))
    pd.DataFrame(sensitivity).to_csv(out/'solar_break_sensitivity.csv',index=False)
    plot_decade(events,centers,sizes,classification,regression,rules,out/'figures')
    write_findings(out,events,classification,regression,rules,improvements,cluster_stats,usable)
    save_json(out/'run_manifest.json',dict(arguments=vars(args),python=platform.python_version(),
         completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),train_rows=len(train),test_rows=len(test),
         protocol_sha256=hashlib.sha256((out/'frozen_protocol.json').read_bytes()).hexdigest(),
         complete_timestamp_coverage=bool(audit.missing.eq(0).all())))
    print('Completed: '+str(out/'pilot_findings.md'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--cache',default='data/cache')
    parser.add_argument('--output',default='results/decade')
    parser.add_argument('--min-support',type=float,default=.02)
    parser.add_argument('--bootstrap',type=int,default=400)
    args=parser.parse_args()
    if args.bootstrap<20:
        parser.error('--bootstrap must be at least 20')
    try:
        with threadpool_limits(limits=2):
            run(args)
    except Exception as error:
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        save_json(out/'failure.json',dict(error=type(error).__name__,message=str(error),arguments=vars(args)))
        raise
