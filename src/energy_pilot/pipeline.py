"""Reproducible CLI and train/test orchestration."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .api import Client, PRODUCTION, PRICES
from .data import Features, clean_join, coverage, FEATURES
from .evaluation import fit_clusters, bootstrap_lift, calendar_comparison
from .mining import apriori, baskets, score_rule
from .plots import figures


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')


def markdown_table(frame):
    """Readable Markdown without an extra tabulate dependency."""
    def cell(value):
        if isinstance(value,(float,np.floating)):
            return f'{value:.4f}' if np.isfinite(value) else 'undefined'
        return str(value).replace('|','\\|').replace('\n',' ')
    rows=['| '+' | '.join(map(str,frame.columns))+' |',
          '| '+' | '.join(['---']*len(frame.columns))+' |']
    rows += ['| '+' | '.join(cell(value) for value in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def acquire(client, start, end, year=None):
    production = client.year(PRODUCTION,year) if year else client.window(PRODUCTION,start,end)
    prices = client.year(PRICES,year) if year else client.window(PRICES,start,end)
    audits = pd.concat([coverage(production,start,end,PRODUCTION),coverage(prices,start,end,PRICES)])
    frame, merge = clean_join(production,prices)
    return frame, audits, merge


def select_rules(train, min_support):
    transaction = baskets(train)
    counts = apriori(transaction,min_support=min_support)
    all_scores, selected = [], []
    for event in ['negative_price','high_price']:
        eligible = []
        for antecedent,count in sorted(counts.items(),key=lambda x: tuple(sorted(x[0]))):
            # Area-qualified cluster rules have interpretable denominators in both years.
            if not any(x.startswith('cluster=') for x in antecedent) or not any(x.startswith('area=') for x in antecedent):
                continue
            score = score_rule(train,transaction,antecedent,event)
            all_scores.append(dict(event=event,antecedent=' & '.join(sorted(antecedent)),**score))
            if score['n'] >= 100 and score['events'] >= 20:
                eligible.append((antecedent,score))
        # Two rules per outcome, prefer lift, then count; one per area if available.
        eligible.sort(key=lambda x: (-x[1]['lift'],-x[1]['n'],tuple(sorted(x[0]))))
        for area in ['DK1','DK2']:
            matches=[item for item in eligible if f'area={area}' in item[0]]
            if matches: selected.append((event,matches[0][0]))
    return selected, pd.DataFrame(all_scores), len(counts)


def analyze(frame, out, train_year=2023, test_year=2024, min_support=.02, bootstrap=400, solar_total=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    train_raw, test_raw = frame[frame.year==train_year], frame[frame.year==test_year]
    if len(train_raw) < 100 or len(test_raw) < 100:
        raise ValueError('Insufficient rows in training or validation year')
    transform=Features().fit(train_raw)
    train,test=transform.transform(train_raw,solar_total),transform.transform(test_raw,solar_total)
    # For sensitivity, use the same rows in both variants: fail rather than hide exclusions.
    if not np.isfinite(pd.concat([train,test])[FEATURES]).all().all():
        raise ValueError('Nonfinite features (including missing estimated self-consumption)')
    profile_columns=['wind_mwh','solar_grid_mwh','solar_total_est_mwh','gross_load_mwh','SpotPriceEUR']+FEATURES
    profiles=[]
    for (year,area),group in pd.concat([train,test]).groupby(['year','PriceArea']):
        for column in profile_columns:
            values=group[column]
            profiles.append(dict(year=year,area=area,variable=column,n=len(values),
                                 missing=int(values.isna().sum()),zeros=int((values==0).sum()),
                                 minimum=values.min(),p01=values.quantile(.01),median=values.median(),
                                 p99=values.quantile(.99),maximum=values.max()))
    pd.DataFrame(profiles).to_csv(out/'quality_profiles.csv',index=False)
    print(f'Training {train_year}: {len(train)}; validation {test_year}: {len(test)}',flush=True)
    train,test,diagnostics,centers,scaler,model=fit_clusters(train,test)
    diagnostics.to_csv(out/'k_diagnostics.csv',index=False)
    centers.to_csv(out/'cluster_centers.csv')
    labels={c:[] for c in centers.index}
    for feature,extreme,label in [('wind_to_load','idxmax','highest wind/load'),
                                  ('solar_to_load','idxmax','highest grid-solar/load'),
                                  ('load_index','idxmin','lowest load index'),
                                  ('load_index','idxmax','highest load index')]:
        labels[getattr(centers[feature],extreme)()].append(label)
    descriptions={int(c):', '.join(tags) if tags else 'intermediate energy regime' for c,tags in labels.items()}
    centers.assign(description=pd.Series(descriptions)).to_csv(out/'cluster_profiles.csv')
    train.groupby(['cluster','PriceArea']).size().rename('n').to_csv(out/'cluster_sizes.csv')
    selected,candidates,itemset_count=select_rules(train,min_support)
    candidates.to_csv(out/'candidate_rules_2023.csv',index=False)
    # Persist the frozen selection before accessing holdout outcome scores.
    save_json(out/'frozen_training.json',dict(train_year=train_year,test_year=test_year,
              medians=transform.medians,quantiles={k:v.tolist() for k,v in transform.quantiles.items()},thresholds=transform.thresholds,
              scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
              centers_scaled=model.cluster_centers_.tolist(),
              rules=[dict(event=e,antecedent=sorted(a)) for e,a in selected],
              min_support=min_support,itemset_count=itemset_count,random_seed=42))
    train_baskets,test_baskets=baskets(train),baskets(test)
    rows=[]
    for event,antecedent in selected:
        a=score_rule(train,train_baskets,antecedent,event)
        b=score_rule(test,test_baskets,antecedent,event)
        ci=bootstrap_lift(test,test_baskets,antecedent,event,repeats=bootstrap)
        rows.append(dict(event=event,antecedent=' & '.join(sorted(antecedent)),
                         **{'train_'+k:v for k,v in a.items()}, **{'test_'+k:v for k,v in b.items()},
                         ci_low=ci['low'],ci_high=ci['high'],bootstrap_valid=ci['valid']))
    rule_columns=['event','antecedent']+[prefix+key for prefix in ['train_','test_']
                  for key in ['n','events','support','joint_support','confidence','base_rate','lift','small_sample']]
    rules=pd.DataFrame(rows,columns=rule_columns+['ci_low','ci_high','bootstrap_valid'])
    rules.to_csv(out/'selected_rules.csv',index=False)
    baseline=pd.DataFrame([r for event in ['negative_price','high_price'] for r in calendar_comparison(train,test,event)])
    baseline.to_csv(out/'calendar_baseline.csv',index=False)
    events=pd.concat([train,test]).groupby(['year','PriceArea']).agg(n=('SpotPriceEUR','size'),
                  negative_events=('negative_price','sum'),high_events=('high_price','sum')).reset_index()
    events['negative_rate']=events.negative_events/events.n
    events['high_rate']=events.high_events/events.n
    events.to_csv(out/'event_rates.csv',index=False)
    figures(train,test,centers,rules,out/'figures')
    pd.concat([train,test]).to_csv(out/'processed.csv.gz',index=False)
    return dict(train=train,test=test,rules=rules,events=events,baseline=baseline,centers=centers,
                bootstrap=bootstrap,descriptions=descriptions,diagnostics=diagnostics)


def findings(result, out, complete):
    r=result
    rule_readout=r['rules'].copy()
    if not rule_readout.empty:
        rule_readout['2023 events/hours']=[f'{int(e)}/{int(n)}' for e,n in zip(rule_readout.train_events,rule_readout.train_n)]
        rule_readout['2024 events/hours']=[f'{int(e)}/{int(n)}' for e,n in zip(rule_readout.test_events,rule_readout.test_n)]
        rule_readout['2024 interval']=[f'[{low:.2f}, {high:.2f}]' for low,high in zip(rule_readout.ci_low,rule_readout.ci_high)]
    lines=['# Pilot findings', '',f'Scope: {"complete requested source coverage" if complete else "INCOMPLETE coverage: inspect coverage.csv; not a complete-year study"}.',
           'Historical associations in realized settlement data; not causal or day-ahead forecasts.','',
           '## Event denominators',markdown_table(r['events']),'','## Frozen rules',
           'No rule passed the training sample requirements.' if r['rules'].empty else markdown_table(rule_readout[[
               'event','antecedent','2023 events/hours','train_lift','2024 events/hours','test_lift','2024 interval','test_small_sample']]),
           '', f'Rules were selected only on 2023, at most one per area and event; exploratory intervals use {r["bootstrap"]} synchronous seven-day moving-block replicates.',
           'Transfer is assessed by validation lift: >1 indicates enrichment against the matching 2024 area base rate; intervals crossing 1 do not clearly support enrichment.',
           '', '## Calendar-only comparison', markdown_table(r['baseline']),
           'Lower holdout Brier score is better. Calendar+cluster uses the same season/time/area groups plus energy cluster; all cell probabilities fit on training only.', '']
    for event,g in r['baseline'].groupby('event'):
        values=g.set_index('model').brier
        gain=values['calendar']-values['calendar_plus_cluster']
        lines.append(f'{event}: calendar+cluster {"improves" if gain>0 else "worsens"} holdout Brier by {abs(gain):.6f}. This is a descriptive probability baseline, not an operational forecast.')
    chosen=r['diagnostics'][r['diagnostics'].k==len(r['centers'])].iloc[0]
    lines += ['', '## Energy regimes (training centroids)',
              f'Selected k={int(chosen.k)}: sampled silhouette {chosen.silhouette:.3f}, Davies-Bouldin {chosen.davies_bouldin:.3f}, smallest cluster {chosen.min_cluster_fraction:.1%}, two-seed adjusted Rand agreement {chosen.seed_stability_ari:.3f}. Silhouette indicates imperfect separation; labels are exploratory summaries.',
              markdown_table(r['centers'].reset_index()),
              '\n'.join(f'C{c}: {description} (relative to the other learned centroids).' for c,description in r['descriptions'].items())]
    if not r['rules'].empty:
        for row in r['rules'].itertuples():
            direction = 'point-estimate enrichment persists' if row.test_lift > 1 else 'point-estimate enrichment does not persist'
            lines.append(f'{row.event}, {row.antecedent}: {direction}; validation lift {row.test_lift:.2f}, exploratory interval [{row.ci_low:.2f}, {row.ci_high:.2f}].')
        intervals=r['rules']
        supported=int(((intervals.ci_low>1)&(intervals.ci_high>1)).sum())
        lines.append(f'{supported}/{len(intervals)} selected rules have exploratory intervals entirely above lift 1; point estimates alone do not establish stable enrichment.')
        negative=r['rules'][r['rules'].event=='negative_price']
        lines += ['',f'Negative-price support: {len(negative)} selected rules; small-sample flags apply to {int(negative.test_small_sample.sum())} (<100 matching hours or <20 matching events in validation). See exact counts above.']
    sensitivity_path=Path(out)/'sensitivity_comparison.csv'
    if sensitivity_path.exists():
        sensitivity=pd.read_csv(sensitivity_path)
        lines += ['', '## Estimated solar self-consumption sensitivity',
                  markdown_table(sensitivity[['event','antecedent','primary_lift_same_rows','lift','lift_difference','n','events']]),
                  'Primary fixed centroids/rules are evaluated with grid solar and total estimated solar on the same available holdout rows. Positive lift_difference means higher enrichment under the estimated total. This is not evidence that the estimate is more accurate.']
    lines += ['', '## Interpretation and limits',
              'Centroids are given in cluster_centers.csv (wind/load, grid-solar/load, area-median load index). Cluster IDs are algorithmic, with no implied ordering.',
              'Gross consumption is a settlement-derived proxy. Grid solar excludes estimated self-consumption. Source metadata warns some reported zeros may represent missing data; zeros are retained as reported, not imputed. This ambiguity cannot be resolved from these sources.',
              'Serial dependence, overlapping rules, training selection and multiple comparisons limit inference. Bootstrap intervals are exploratory and do not correct discovery bias. Repeat with different block lengths before final reporting.',
              'Generation is observed after market clearing. Interconnectors, fuel costs and weather are omitted; no causal interpretation is supported.',
              'Sensitivity results (separately refitted on training only) are in sensitivity/ and sensitivity_comparison.csv; IDs across fits are not equivalent.',
              '', '## Next iteration',
              '1. Inspect data-quality and solar-sensitivity diagnostics before interpreting rules.',
              '2. Check block-length and rule-selection stability without tuning against 2024.',
              '3. Verify the final submission date on DTU Learn; plan for December 4, 2026 (the assignment also mentions December 6).',
              '', 'Source: Energinet Energi Data Service, CC BY 4.0. Exact cached request envelopes and metadata: data/cache/.']
    (Path(out)/'pilot_findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--train-year',type=int,default=2023)
    parser.add_argument('--test-year',type=int,default=2024)
    parser.add_argument('--cache',default='data/cache')
    parser.add_argument('--output',default='results')
    parser.add_argument('--min-support',type=float,default=.02)
    parser.add_argument('--bootstrap',type=int,default=400)
    args=parser.parse_args()
    if args.bootstrap < 20 and not args.smoke_test:
        parser.error('--bootstrap must be >=20 for the main pilot')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    client=Client(args.cache,args.offline)
    try:
        for dataset in [PRODUCTION,PRICES]: client.metadata(dataset)
        if args.smoke_test:
            frame,audit,merge=acquire(client,'2024-01-01','2024-01-08')
            audit.to_csv(out/'smoke_coverage.csv',index=False)
            save_json(out/'smoke_merge.json',merge)
            print(json.dumps(merge,indent=2)); print(frame.head().to_string(index=False))
            frame.to_csv(out/'smoke_merged.csv',index=False)
            # Vertical slice: exploratory split within this week, never full-year evidence.
            midpoint=frame.HourUTC.min()+pd.Timedelta(days=4)
            demo=frame.copy(); demo.loc[demo.HourUTC<midpoint,'year']=2023; demo.loc[demo.HourUTC>=midpoint,'year']=2024
            analyze(demo,out/'smoke_only',bootstrap=10,min_support=.02)
            print('Smoke vertical slice executed; temporary split is not a year-to-year result.')
            return
        if (args.train_year,args.test_year)!=(2023,2024):
            raise ValueError('Pilot scope is fixed to training 2023 / validation 2024')
        frames,coverage_rows,merges=[],[],{}
        for year in [args.train_year,args.test_year]:
            frame,audit,merge=acquire(client,f'{year}-01-01',f'{year+1}-01-01',year)
            frames.append(frame); coverage_rows.append(audit); merges[year]=merge
        audit=pd.concat(coverage_rows)
        audit.to_csv(out/'coverage.csv',index=False); save_json(out/'merge_audit.json',merges)
        frame=pd.concat(frames,ignore_index=True)
        result=analyze(frame,out,args.train_year,args.test_year,args.min_support,args.bootstrap)
        complete=bool((audit.missing==0).all() and all(m['join_retention_fraction']==1 for m in merges.values()))
        sensitivity_frame=frame[np.isfinite(frame.solar_total_est_mwh)].copy()
        if len(sensitivity_frame)!=len(frame):
            save_json(out/'sensitivity_exclusions.json',dict(excluded=len(frame)-len(sensitivity_frame)))
        sensitivity=analyze(sensitivity_frame,out/'sensitivity',min_support=args.min_support,bootstrap=args.bootstrap,solar_total=True)
        # Same selected primary rules under fixed primary centroids: isolate solar feature perturbation.
        frozen=json.loads((out/'frozen_training.json').read_text())
        transform=Features().fit(frame[frame.year==2023])
        altered=transform.transform(sensitivity_frame[sensitivity_frame.year==2024],solar_total=True)
        scaled=(altered[FEATURES].to_numpy()-np.array(frozen['scaler_mean']))/np.array(frozen['scaler_scale'])
        altered['cluster']=((scaled[:,None,:]-np.array(frozen['centers_scaled'])[None,:,:])**2).sum(axis=2).argmin(axis=1)
        primary_same=transform.transform(sensitivity_frame[sensitivity_frame.year==2024])
        primary_scaled=(primary_same[FEATURES].to_numpy()-np.array(frozen['scaler_mean']))/np.array(frozen['scaler_scale'])
        primary_same['cluster']=((primary_scaled[:,None,:]-np.array(frozen['centers_scaled'])[None,:,:])**2).sum(axis=2).argmin(axis=1)
        comparison=[]
        for rule in frozen['rules']:
            primary_score=score_rule(primary_same,baskets(primary_same),frozenset(rule['antecedent']),rule['event'])
            changed_score=score_rule(altered,baskets(altered),frozenset(rule['antecedent']),rule['event'])
            comparison.append(dict(event=rule['event'],antecedent=' & '.join(rule['antecedent']),
                                   primary_lift_same_rows=primary_score['lift'],
                                   lift_difference=changed_score['lift']-primary_score['lift'],**changed_score))
        comparison_columns=['event','antecedent','n','events','support','joint_support','confidence','base_rate','lift','small_sample','primary_lift_same_rows','lift_difference']
        pd.DataFrame(comparison,columns=comparison_columns).to_csv(out/'sensitivity_comparison.csv',index=False)
        findings(result,out,complete)
        save_json(out/'run_manifest.json',dict(arguments=vars(args),python=platform.python_version(),
                                             completed_utc=pd.Timestamp.now(tz='UTC').isoformat(),complete_coverage=complete))
        print((out/'pilot_findings.md').read_text(encoding='utf-8'))
    except Exception as error:
        save_json(out/'last_failure.json',dict(type=type(error).__name__,message=str(error),arguments=vars(args)))
        raise
