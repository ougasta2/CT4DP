"""Independent Apriori core: self-join, subset pruning and exact vertical counting."""
from itertools import combinations
from math import ceil

import numpy as np

DIMENSIONS = ['cluster', 'area', 'demand', 'time_of_day', 'season']


def candidates(previous, size):
    previous = set(previous)
    result = set()
    for left, right in combinations(sorted(previous, key=lambda x: sorted(x)), 2):
        union = left | right
        if len(union) != size:
            continue
        # A basket contains at most one item from each categorical dimension.
        if len({x.split('=', 1)[0] for x in union}) != size:
            continue
        if all(frozenset(subset) in previous for subset in combinations(sorted(union), size-1)):
            result.add(union)
    return result


def apriori(baskets, min_support=.02, max_size=3):
    if not baskets or not 0 < min_support <= 1 or not 1 <= max_size <= 3:
        raise ValueError('Require nonempty baskets, support in (0,1], and max_size in 1..3')
    vertical = {}
    for row, basket in enumerate(baskets):
        dimensions = [item.split('=', 1)[0] for item in basket]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError('Multiple items from one dimension')
        for item in basket:
            vertical.setdefault(item, set()).add(row)
    threshold = ceil(min_support * len(baskets))
    level = {frozenset([item]): len(rows) for item, rows in vertical.items() if len(rows) >= threshold}
    frequent = dict(level)
    for size in range(2, max_size+1):
        next_level = {}
        for candidate in candidates(level, size):
            count = len(set.intersection(*(vertical[item] for item in candidate)))
            if count >= threshold:
                next_level[candidate] = count
        frequent.update(next_level)
        level = next_level
    return frequent


def baskets(frame):
    return [frozenset([f'cluster=C{c}', f'area={a}', f'demand={d}', f'time_of_day={t}', f'season={s}'])
            for c,a,d,t,s in zip(frame.cluster, frame.PriceArea, frame.demand, frame.time_of_day, frame.season)]


def mask_for(baskets_, antecedent):
    return np.fromiter((antecedent <= basket for basket in baskets_), dtype=bool, count=len(baskets_))


def score_rule(frame, baskets_, antecedent, event, weights=None):
    mask = mask_for(baskets_, antecedent)
    area = next((x.split('=')[1] for x in antecedent if x.startswith('area=')), None)
    eligible = (frame.PriceArea.to_numpy() == area) if area else np.ones(len(frame), dtype=bool)
    weights = np.ones(len(frame)) if weights is None else np.asarray(weights)
    y = frame[event].to_numpy(dtype=float)
    n, events = float(weights[mask].sum()), float((weights[mask] * y[mask]).sum())
    denominator = weights[eligible].sum()
    base = np.sum(weights[eligible]*y[eligible])/denominator if denominator else np.nan
    rate = events/n if n else np.nan
    return dict(n=n, events=events, support=n/weights.sum(), joint_support=events/weights.sum(),
                confidence=rate, base_rate=base, lift=rate/base if base > 0 else np.nan,
                small_sample=bool(n < 100 or events < 20))
