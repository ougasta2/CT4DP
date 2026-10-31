from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from energy_pilot.api import WIND,SOLAR,SELF
from energy_pilot.data import keyed,coverage,clean_join,Features,TZ
from energy_pilot.mining import apriori,candidates,score_rule
from energy_pilot.evaluation import bootstrap_lift,calendar_comparison


def test_apriori_hand_counts_and_pruning():
    baskets=[frozenset(x) for x in [('a=1','b=1','c=1'),('a=1','b=1'),('a=1','c=1'),('b=1',)]]
    f=apriori(baskets,.5)
    assert f[frozenset(['a=1'])]==3
    assert f[frozenset(['a=1','b=1'])]==2
    assert f[frozenset(['a=1','c=1'])]==2
    assert frozenset(['b=1','c=1']) not in f
    assert candidates([frozenset(['a=1','b=1']),frozenset(['a=1','c=1'])],3)==set()


def test_apriori_against_exhaustive_counts():
    rng=np.random.default_rng(4)
    baskets=[frozenset(f'{d}={rng.integers(2)}' for d in 'abcd') for _ in range(70)]
    result=apriori(baskets,.15)
    universe=sorted(set.union(*map(set,baskets)))
    reference={}
    for k in range(1,4):
        for subset in combinations(universe,k):
            item=frozenset(subset)
            count=sum(item<=b for b in baskets)
            if count>=np.ceil(.15*len(baskets)): reference[item]=count
    assert result==reference


def test_rare_outcome_not_pruned_and_area_base():
    frame=pd.DataFrame({'PriceArea':['DK1']*50+['DK2']*50,'negative_price':[True]+[False]*99})
    baskets=[frozenset(['area='+a,'cluster=C0']) for a in frame.PriceArea]
    antecedent=frozenset(['area=DK1','cluster=C0'])
    assert antecedent in apriori(baskets,.2)
    score=score_rule(frame,baskets,antecedent,'negative_price')
    assert score['events']==1 and score['n']==50
    assert score['joint_support']==.01 and score['lift']==1


def test_dst_leap_and_local_year():
    for day,expected in [('2024-03-31',23),('2024-10-27',25)]:
        start=pd.Timestamp(day,tz=TZ)
        end=pd.Timestamp(start.date()+pd.Timedelta(days=1),tz=TZ)
        assert len(pd.date_range(start,end,freq='h',inclusive='left'))==expected
    index=pd.date_range('2024-01-01','2025-01-01',tz=TZ,freq='h',inclusive='left').tz_convert('UTC')
    assert len(index)==8784 and index[0]==pd.Timestamp('2023-12-31T23:00Z')
    frame=pd.DataFrame({'HourUTC':index.repeat(2),'PriceArea':['DK1','DK2']*len(index)})
    audit=coverage(frame,'2024-01-01','2025-01-01','fixture')
    assert audit.missing.sum()==0 and audit.expected.sum()==17568
    assert index.tz_convert(TZ).year.min()==2024
    repeated=pd.to_datetime(['2024-10-27T00:00Z','2024-10-27T01:00Z'],utc=True)
    assert repeated.tz_convert(TZ).hour.tolist()==[2,2]
    # Both UTC keys survive even though their naive Danish wall times coincide.
    assert len(keyed(pd.DataFrame({'HourUTC':repeated,'PriceArea':['DK1','DK1']})))==2


def fixtures():
    production=pd.DataFrame({'HourUTC':['2023-12-31T23:00:00','2024-01-01T00:00:00'],'PriceArea':['DK1']*2})
    for col in WIND+SOLAR+[SELF]: production[col]=1.
    production['GrossConsumptionMWh']=10.
    prices=production[['HourUTC','PriceArea']].copy(); prices['SpotPriceEUR']=[-10.,50.]
    return production,prices


def test_duplicate_and_merge_audit_missing_not_zero():
    p,q=fixtures()
    with pytest.raises(ValueError,match='duplicate'): keyed(pd.concat([p,p]))
    frame,audit=clean_join(p,q.iloc[:1])
    assert len(frame)==1 and audit['merge_counts']['left_only']==1
    assert frame.year.iloc[0]==2024 and frame.wind_mwh.iloc[0]==4
    p.loc[0,WIND[0]]=np.nan
    frame,audit=clean_join(p,q)
    assert len(frame)==1 and audit['excluded_incomplete_or_invalid_rows']==1


def test_features_frozen_no_clipping():
    p,q=fixtures(); frame,_=clean_join(p,q)
    transform=Features().fit(frame)
    test=frame.copy(); test['wind_mwh']=100.; test['gross_load_mwh']=20.;test['SpotPriceEUR']=1000.
    output=transform.transform(test)
    assert (output.wind_to_load==5).all()
    assert (output.load_index==2).all()
    assert transform.medians=={'DK1':10.}
    assert (output.high_price).all()


def test_block_bootstrap_deterministic_constant_lift():
    hours=pd.date_range('2024-03-20',periods=24*40,freq='h',tz='UTC')
    frame=pd.DataFrame({'PriceArea':'DK1','local_date':hours.tz_convert(TZ).date,'negative_price':True})
    baskets=[frozenset(['area=DK1'])]*len(frame)
    kwargs=dict(frame=frame,baskets=baskets,antecedent=frozenset(['area=DK1']),event='negative_price',repeats=30)
    a=bootstrap_lift(**kwargs); b=bootstrap_lift(**kwargs)
    assert a==b and a['low']==a['high']==1 and a['valid']==30


def test_calendar_unseen_backoff_uses_training_only():
    train=pd.DataFrame({'PriceArea':['DK1']*2,'season':['winter']*2,'time_of_day':['night']*2,'cluster':[0,0],'negative_price':[True,False]})
    test=pd.DataFrame({'PriceArea':['DK1'],'season':['summer'],'time_of_day':['night'],'cluster':[1],'negative_price':[True]})
    result=calendar_comparison(train,test,'negative_price')
    assert all(row['fallback_rows']==1 and row['brier']==.25 for row in result)


def test_no_events_produces_no_invented_shortlist(tmp_path):
    from energy_pilot.pipeline import analyze
    rng=np.random.default_rng(21)
    times=pd.date_range('2023-01-01',periods=200,freq='h',tz='UTC').append(
        pd.date_range('2024-01-01',periods=200,freq='h',tz='UTC'))
    p=pd.DataFrame({'HourUTC':times,'PriceArea':['DK1','DK2']*200})
    for col in WIND+SOLAR+[SELF]: p[col]=rng.uniform(0,20,len(p))
    p['GrossConsumptionMWh']=rng.uniform(50,100,len(p))
    q=p[['HourUTC','PriceArea']].copy(); q['SpotPriceEUR']=10.
    frame,_=clean_join(p,q)
    result=analyze(frame,tmp_path,bootstrap=20)
    assert result['rules'].empty
    assert pd.read_csv(tmp_path/'selected_rules.csv').empty
    assert (tmp_path/'figures/week.png').exists()
    assert (result['baseline'].brier==0).all()
