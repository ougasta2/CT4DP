"""Evaluate the price-regression model's negative-price sign."""
from pathlib import Path
import numpy as np
import pandas as pd


def write_sign_metrics(predictions,out):
    frame=predictions.copy();frame['observed_negative']=frame.actual<0;frame['predicted_negative']=frame.prediction<0
    frame['true_positive']=frame.observed_negative & frame.predicted_negative
    result=frame.groupby(['zone','model']).agg(n=('actual','size'),actual_negative=('observed_negative','sum'),
        predicted_negative=('predicted_negative','sum'),true_positive=('true_positive','sum')).reset_index()
    result['recall']=np.where(result.actual_negative>0,result.true_positive/result.actual_negative,np.nan)
    result['precision']=np.where(result.predicted_negative>0,result.true_positive/result.predicted_negative,np.nan)
    result.to_csv(Path(out)/'negative_sign_metrics.csv',index=False)


if __name__=='__main__':
    out=Path('results/price_forecast')
    write_sign_metrics(pd.read_csv(out/'predictions.csv.gz'),out)
