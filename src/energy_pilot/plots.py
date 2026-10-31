"""Static figures with physical units and explicit event denominators."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def figures(train, test, centers, rules, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    week = test[(test.HourUTC >= test.HourUTC.min()) & (test.HourUTC < test.HourUTC.min() + __import__('pandas').Timedelta(days=7))]
    fig, axes = plt.subplots(2,2,figsize=(12,6),sharex=True,sharey='col',layout='constrained')
    for i, area in enumerate(['DK1','DK2']):
        g = week[week.PriceArea == area]
        axes[i,0].plot(g.HourUTC,g.wind_mwh,label='Wind')
        axes[i,0].plot(g.HourUTC,g.solar_grid_mwh,label='Solar to grid')
        axes[i,0].plot(g.HourUTC,g.gross_load_mwh,label='Gross load',linestyle='--')
        axes[i,0].set(ylabel=f'{area}: MWh / hour interval')
        axes[i,1].plot(g.HourUTC,g.SpotPriceEUR,color='#7030a0')
        axes[i,1].axhline(0,color='gray',linewidth=.7)
        axes[i,1].set(ylabel=f'{area}: EUR/MWh')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('First validation week (preselected): realized energy and day-ahead price')
    for ax in axes[-1]: ax.set_xlabel('Hour (UTC)')
    fig.autofmt_xdate()
    fig.savefig(out/'week.png',dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained',sharex=True,sharey=True)
    for ax,area in zip(axes,['DK1','DK2']):
        g = test[test.PriceArea == area]
        h = ax.hexbin(g.wind_to_load,g.solar_to_load,C=g.negative_price.astype(float),
                      reduce_C_function=np.mean,gridsize=22,mincnt=20,vmin=0,vmax=1,cmap='viridis')
        ax.set(title=f'{area}, validation ({int(g.year.iloc[0])})',xlabel='Wind / gross load (MWh/MWh)')
    axes[0].set_ylabel('Solar to grid / gross load (MWh/MWh)')
    fig.colorbar(h,ax=axes,label='Negative-price fraction (cells with >20 hours)')
    fig.savefig(out/'regimes.png',dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    centers.plot.bar(ax=axes[0]); axes[0].set(title='2023-trained centroids',ylabel='Dimensionless feature value')
    axes[0].legend(fontsize=8)
    train.groupby(['cluster','PriceArea']).size().unstack(fill_value=0).plot.bar(ax=axes[1])
    axes[1].set(title='Training cluster membership',ylabel='Area-hours')
    fig.savefig(out/'clusters.png',dpi=160); plt.close(fig)

    if len(rules):
        fig,ax = plt.subplots(figsize=(10,max(4,len(rules)*.75)),layout='constrained')
        y=np.arange(len(rules))
        ax.scatter(rules.train_lift,y-.12,label='2023 discovery',marker='s')
        ax.scatter(rules.test_lift,y+.12,label='2024 validation',marker='o')
        for i,r in rules.iterrows():
            if np.isfinite(r.ci_low) and np.isfinite(r.ci_high):
                ax.plot([r.ci_low,r.ci_high],[i+.12,i+.12],color='#ff7f0e')
        ax.axvline(1,color='gray',linestyle='--',label='Area base rate')
        ax.set_yticks(y,[f"{r.event}: {r.antecedent}\n2024 events/hours: {int(r.test_events)}/{int(r.test_n)}" for _,r in rules.iterrows()],fontsize=8)
        ax.set(xlabel='Event-rate lift vs same-year area baseline',title='Fixed 2023 rules; exploratory 95% seven-day bootstrap intervals')
        ax.legend(fontsize=8)
        fig.savefig(out/'rules.png',dpi=160); plt.close(fig)
