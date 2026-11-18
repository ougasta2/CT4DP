"""Equal-weight ensemble of aligned team predictions; no weights are learned."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def combine(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=['HourUTC', 'day'])
        required = {'HourUTC', 'day', 'hour', 'actual', 'prediction'}
        if not required.issubset(frame) or frame.empty or frame.HourUTC.duplicated().any():
            raise ValueError(f'Invalid prediction file: {path}')
        frame = frame.sort_values('HourUTC').reset_index(drop=True)
        if frame[list(required)].isna().any().any() or not np.isfinite(frame[['actual','prediction']].to_numpy()).all():
            raise ValueError(f'Missing or non-finite values: {path}')
        frames.append(frame)
    if len(frames) < 2:
        raise ValueError('Provide at least two prediction files')
    for f in frames[1:]:
        if not frames[0][['HourUTC', 'day', 'hour', 'actual']].equals(f[['HourUTC', 'day', 'hour', 'actual']]):
            raise ValueError('Prediction files do not have exactly matching timestamps, targets and actuals')
    out = frames[0][['HourUTC', 'day', 'hour', 'actual']].copy()
    out['prediction'] = np.mean([f.prediction.to_numpy() for f in frames], axis=0)
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/team/ensemble/predictions.csv.gz')
    args = parser.parse_args()
    result = combine(args.files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, compression={'method':'gzip', 'mtime':0} if str(args.output).endswith('.gz') else None)
    rows = []
    for year, group in result.groupby(result.day.dt.year):
        error = (group.prediction - group.actual).abs()
        eligible = group.actual.abs() >= 20
        negative = group.actual < 0
        rows.append({'year': int(year), 'hours': len(group), 'mae': error.mean(),
                     'rmse': np.sqrt((error ** 2).mean()), 'eligible': int(eligible.sum()),
                     'hits_5pct': int((error[eligible] <= .05 * group.actual[eligible].abs()).sum()),
                     'rate_5pct': (error[eligible] <= .05 * group.actual[eligible].abs()).mean() if eligible.any() else np.nan,
                     'negative_hours': int(negative.sum()), 'negative_mae': error[negative].mean() if negative.any() else np.nan})
    metrics_path = args.output.parent / 'metrics.csv'
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f'Wrote {len(result)} aligned predictions to {args.output} and metrics to {metrics_path}')

if __name__ == '__main__':
    main()
