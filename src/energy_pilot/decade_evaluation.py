"""Comparable out-of-time classification/regression scores and day-block rules."""
import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss,average_precision_score,roc_auc_score,
                             precision_score,recall_score,mean_absolute_error,mean_squared_error)
from .mining import mask_for,score_rule


def groups(test):
    yield '2024-2025','both',np.ones(len(test),dtype=bool)
    for year in sorted(test.year.unique()):
        yield str(year),'both',(test.year==year).to_numpy()
        for area in ['DK1','DK2']:
            yield str(year),area,((test.year==year)&(test.PriceArea==area)).to_numpy()


def classification_metrics(test,event,probability,method,features):
    probability=np.asarray(probability)
    if not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():
        raise ValueError('Invalid class probabilities')
    rows=[]
    for year,area,keep in groups(test):
        y=test.loc[keep,event].to_numpy(dtype=int); p=probability[keep]
        both=len(np.unique(y))==2
        rows.append(dict(method=method,features=features,event=event,test_year=year,area=area,
                         n=len(y),events=int(y.sum()),base_rate=float(y.mean()),
                         brier=float(brier_score_loss(y,p)),
                         average_precision=float(average_precision_score(y,p)) if y.sum() else np.nan,
                         roc_auc=float(roc_auc_score(y,p)) if both else np.nan,
                         precision_at_0_5=float(precision_score(y,p>=.5,zero_division=0)),
                         recall_at_0_5=float(recall_score(y,p>=.5,zero_division=0))))
    return rows


def regression_metrics(test,prediction,method,features):
    prediction=np.asarray(prediction)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite regression predictions')
    rows=[]
    for year,area,keep in groups(test):
        y=test.loc[keep,'SpotPriceEUR'].to_numpy(); p=prediction[keep]
        rows.append(dict(method=method,features=features,test_year=year,area=area,n=len(y),
                         mae_eur_mwh=float(mean_absolute_error(y,p)),
                         rmse_eur_mwh=float(np.sqrt(mean_squared_error(y,p))),
                         mean_error_eur_mwh=float(np.mean(p-y))))
    return rows


def cell_predictions(train,test,target,cluster=False,median=False):
    keys=['PriceArea','season','time_of_day']+(['cluster'] if cluster else [])
    operation='median' if median else 'mean'
    rates=getattr(train.groupby(keys,observed=True)[target],operation)()
    index=pd.MultiIndex.from_frame(test[keys])
    result=pd.Series(rates.reindex(index).to_numpy(),index=test.index)
    area=getattr(train.groupby('PriceArea')[target],operation)()
    pooled=getattr(train[target],operation)()
    fallback=int(result.isna().sum())
    result=result.fillna(test.PriceArea.map(area)).fillna(pooled)
    return result.to_numpy(),fallback


def fast_block_interval(frame,baskets,antecedent,event,repeats=400,block_days=7,seed=42):
    """Equivalent block resampling via sufficient daily counts (both areas synchronous)."""
    dates=pd.date_range(min(frame.local_date),max(frame.local_date),freq='D').date
    if len(dates)<block_days*2:
        return dict(low=np.nan,high=np.nan,valid=0,repeats=repeats)
    rule=mask_for(baskets,antecedent)
    area=next((item.split('=')[1] for item in antecedent if item.startswith('area=')),None)
    eligible=frame.PriceArea.eq(area).to_numpy() if area else np.ones(len(frame),dtype=bool)
    target=frame[event].to_numpy(dtype=int)
    daily=pd.DataFrame({'date':frame.local_date,'rule_n':rule.astype(int),
                        'rule_events':rule*target,'base_n':eligible.astype(int),'base_events':eligible*target})
    counts=daily.groupby('date').sum().reindex(dates,fill_value=0).to_numpy()
    rng=np.random.default_rng(seed); values=[]
    for _ in range(repeats):
        starts=rng.integers(0,len(dates)-block_days+1,size=int(np.ceil(len(dates)/block_days)))
        selected=np.concatenate([np.arange(start,start+block_days) for start in starts])[:len(dates)]
        n,events,total,base_events=counts[selected].sum(axis=0)
        if n>0 and total>0 and base_events>0:
            values.append(float((events/n)/(base_events/total)))
    low,high=np.quantile(values,[.025,.975]) if len(values)>=repeats*.8 else (np.nan,np.nan)
    return dict(low=float(low),high=float(high),valid=len(values),repeats=repeats)


def evaluate_rules(train,test,selected,method,repeats):
    from .mining import baskets
    train_baskets=baskets(train); test_baskets=baskets(test)
    rows=[]; yearly=[]
    for event,antecedent in selected:
        a=score_rule(train,train_baskets,antecedent,event)
        b=score_rule(test,test_baskets,antecedent,event)
        interval=fast_block_interval(test,test_baskets,antecedent,event,repeats)
        rows.append(dict(method=method,event=event,antecedent=' & '.join(sorted(antecedent)),
                         **{'train_'+k:v for k,v in a.items()},**{'test_'+k:v for k,v in b.items()},
                         ci_low=interval['low'],ci_high=interval['high'],bootstrap_valid=interval['valid']))
        for year in sorted(test.year.unique()):
            g=test[test.year==year]
            yearly.append(dict(method=method,event=event,antecedent=' & '.join(sorted(antecedent)),year=int(year),
                               **score_rule(g,baskets(g),antecedent,event)))
    return rows,yearly


def paired_error_interval(test,baseline_errors,model_errors,repeats=400,block_days=7):
    """Positive delta = improvement, paired synchronous day-block resampling."""
    dates=pd.date_range(min(test.local_date),max(test.local_date),freq='D').date
    frame=pd.DataFrame({'date':test.local_date,'delta':np.asarray(baseline_errors)-np.asarray(model_errors),'n':1})
    counts=frame.groupby('date')[['delta','n']].sum().reindex(dates,fill_value=0).to_numpy()
    rng=np.random.default_rng(42);samples=[]
    for _ in range(repeats):
        starts=rng.integers(0,len(dates)-block_days+1,size=int(np.ceil(len(dates)/block_days)))
        index=np.concatenate([np.arange(s,s+block_days) for s in starts])[:len(dates)]
        delta,n=counts[index].sum(axis=0)
        samples.append(delta/n)
    low,high=np.quantile(samples,[.025,.975])
    return dict(improvement=float(frame.delta.mean()),ci_low=float(low),ci_high=float(high),
                repeats=repeats,block_days=block_days)
