"""Figures for the extended temporal benchmark; each panel keeps its own metric."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_decade(events,centers,sizes,class_scores,reg_scores,rules,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,event,title in zip(axes,['negative_rate','high_rate'],['Negative hourly price','Above training area 95th percentile']):
        for area,marker in [('DK1','o'),('DK2','s')]:
            g=events[events.PriceArea==area]
            ax.plot(g.year,100*g[event],marker=marker,label=area)
        ax.axvspan(2023.5,2025.5,color='gray',alpha=.15,label='Holdout')
        ax.set(xlabel='Copenhagen calendar year',ylabel='Area-hours (%)',title=title,xticks=[2016,2018,2020,2022,2024,2025])
    axes[0].legend(fontsize=8)
    fig.savefig(output/'yearly_events.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    g=centers[centers.method=='kmeans'].set_index('cluster')
    g[['wind_to_load','solar_to_load','load_index']].rename(columns={'solar_to_load':'harmonized solar/load'}).plot.bar(ax=axes[0])
    axes[0].set(title='K-means: 2016–2023 training centroids',ylabel='Dimensionless feature value')
    axes[0].legend(fontsize=8)
    sizes[sizes.method=='kmeans'].pivot(index='year',columns='cluster',values='fraction').plot.area(ax=axes[1],colormap='tab10')
    axes[1].axvline(2023.5,color='black',linestyle='--');axes[1].set(title='Frozen K-means membership by year',ylabel='Fraction of area-hours',ylim=(0,1))
    axes[1].legend(title='Cluster',fontsize=8,ncol=2)
    fig.savefig(output/'clusters_and_drift.png',dpi=160);plt.close(fig)

    c=class_scores[(class_scores.test_year=='2024-2025')&(class_scores.area=='both')].copy()
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,event in zip(axes,['negative_price','high_price']):
        g=c[c.event==event].copy()
        g['label']=g.method+' / '+g.features
        g=g.sort_values('brier',ascending=True)
        ax.barh(g.label,g.brier,color='#3274a1')
        ax.invert_yaxis();ax.set(title=event.replace('_',' '),xlabel='Holdout Brier score (lower is better)')
        ax.tick_params(axis='y',labelsize=8)
    fig.savefig(output/'classification_comparison.png',dpi=160);plt.close(fig)

    r=reg_scores[(reg_scores.test_year=='2024-2025')&(reg_scores.area=='both')].copy()
    r['label']=[('Boosting / ' if method=='gradient_boosting' else 'Calendar cells / ')+
                ('energy + calendar' if features=='energy_calendar' else 'calendar') for method,features in zip(r.method,r.features)]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric,label in zip(axes,['mae_eur_mwh','rmse_eur_mwh'],['MAE (EUR/MWh)','RMSE (EUR/MWh)']):
        g=r.sort_values(metric)
        ax.barh(g.label,g[metric],color='#397d49');ax.invert_yaxis()
        ax.set(xlabel=label,title=label.split(' (')[0]);ax.tick_params(axis='y',labelsize=9)
    fig.suptitle('2024–2025 price regression (lower is better)')
    fig.savefig(output/'regression_comparison.png',dpi=160);plt.close(fig)

    if len(rules):
        selected=rules[rules.method=='kmeans'].reset_index(drop=True)
        fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
        y=np.arange(len(selected))
        ax.scatter(selected.train_lift,y-.12,marker='s',label='2016–2023 training')
        ax.scatter(selected.test_lift,y+.12,marker='o',label='2024–2025 test')
        for i,row in selected.iterrows():
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                ax.plot([row.ci_low,row.ci_high],[i+.12,i+.12],color='#ff7f0e')
        ax.axvline(1,color='gray',linestyle='--')
        ax.set_yticks(y,[f'{r.event}: {r.antecedent}\nevents/hours: {int(r.test_events)}/{int(r.test_n)}' for r in selected.itertuples()],fontsize=8)
        ax.set(xlabel='Lift against matching test-area baseline',title='K-means + Apriori: fixed rules, exploratory 95% seven-day block intervals')
        ax.legend(fontsize=8)
        fig.savefig(output/'frozen_rules.png',dpi=160);plt.close(fig)
