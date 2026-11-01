"""Additional fixed pilot methods; train-only fitting and frozen core extension."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor

from .data import FEATURES

SEED=42
RF_CONFIG=dict(n_estimators=120,max_depth=12,min_samples_leaf=80,max_features=1.0,
               n_jobs=2,random_state=SEED)
REG_CONFIG=dict(max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=100,
                l2_regularization=1.,early_stopping=False,random_state=SEED)


def sampled_silhouette(values,labels,maximum=2000):
    """A random metric sample can contain only one of several very tiny clusters."""
    index=np.random.RandomState(SEED).permutation(len(values))[:maximum]
    observed=np.asarray(labels)[index]
    if not 2<=len(np.unique(observed))<len(index):
        return np.nan
    return float(silhouette_score(np.asarray(values)[index],observed))


def feature_matrix(frame,energy=True):
    """No year, target, future quantities, or price-derived features."""
    out=pd.DataFrame(index=frame.index)
    out['area_DK2']=(frame.PriceArea=='DK2').astype(float)
    for name,values,period in [('hour',frame.local_hour,24),('month',frame.month-1,12),
                                ('weekday',frame.weekday,7)]:
        out[name+'_sin']=np.sin(2*np.pi*values/period)
        out[name+'_cos']=np.cos(2*np.pi*values/period)
    if energy:
        out=out.join(frame[FEATURES])
    if not np.isfinite(out).all().all():
        raise ValueError('Supervised inputs must be finite')
    return out


def stratified_sample(frame,maximum=20000):
    """Deterministic representative sample from every training year and area."""
    if len(frame)<=maximum:
        return np.arange(len(frame))
    groups=list(frame.groupby(['year','PriceArea'],sort=True).indices.values())
    per=max(1,maximum//len(groups))
    rng=np.random.default_rng(SEED)
    return np.sort(np.concatenate([rng.choice(g,size=min(per,len(g)),replace=False) for g in groups]))


class CoreAssignment:
    """DBSCAN has no native predict: assign only within eps of a frozen core.

    Points outside the training core neighborhoods remain noise (-1). This is an
    explicitly defined inductive adapter, not refitting DBSCAN on holdout rows.
    """
    def __init__(self,model):
        self.eps=model.eps
        self.core_labels=model.labels_[model.core_sample_indices_]
        self.neighbors=NearestNeighbors(n_neighbors=1).fit(model.components_) if len(model.components_) else None

    def predict(self,values):
        labels=np.full(len(values),-1,dtype=int)
        if self.neighbors is not None:
            distance,index=self.neighbors.kneighbors(values)
            keep=distance[:,0]<=self.eps
            labels[keep]=self.core_labels[index[keep,0]]
        return labels


def clustering_extensions(train,test,scaler):
    x=scaler.transform(train[FEATURES]); z=scaler.transform(test[FEATURES])
    diagnostics=[]
    models=[]
    for k in range(3,8):
        model=GaussianMixture(n_components=k,covariance_type='full',reg_covar=1e-5,
                              n_init=2,max_iter=200,random_state=SEED).fit(x)
        labels=model.predict(x)
        smallest=np.bincount(labels,minlength=k).min()/len(x)
        diagnostics.append(dict(method='gmm',parameter=k,bic=model.bic(x),converged=model.converged_,
                                 min_cluster_fraction=smallest,noise_fraction=0.))
        if model.converged_ and smallest>=.03:
            models.append((model.bic(x),model))
    if not models:
        raise ValueError('No converged GMM candidate with all components >=3%')
    gmm=min(models,key=lambda pair:pair[0])[1]
    result={'gmm':(gmm.predict(x),gmm.predict(z),gmm)}
    sample_index=stratified_sample(train)
    sample=x[sample_index]
    neighbors=NearestNeighbors(n_neighbors=min(20,len(sample))).fit(sample)
    radii=neighbors.kneighbors(sample)[0][:,-1]
    options=[]
    for quantile in [.65,.8,.95]:
        eps=float(max(np.quantile(radii,quantile),1e-4))
        model=DBSCAN(eps=eps,min_samples=20,n_jobs=2).fit(sample)
        assigned=model.labels_
        keep=assigned>=0
        count=len(set(assigned[keep]))
        noise=float((~keep).mean())
        silhouette=sampled_silhouette(sample[keep],assigned[keep],1500) if count>=2 and keep.sum()>count else np.nan
        diagnostics.append(dict(method='dbscan',parameter=eps,k_distance_quantile=quantile,
                                 sample_rows=len(sample),clusters=count,noise_fraction=noise,
                                 silhouette=silhouette))
        # No test events are consulted. Failed separation remains visible.
        objective=silhouette-.2*noise if count>=2 and noise<=.5 and np.isfinite(silhouette) else -2-noise
        options.append((objective,-eps,model))
    dbscan=max(options,key=lambda option:option[:2])[2]
    extension=CoreAssignment(dbscan)
    result['dbscan']=(extension.predict(x),extension.predict(z),extension)
    return result,pd.DataFrame(diagnostics),sample_index


def cluster_diagnostics(x,labels):
    labels=np.asarray(labels)
    keep=labels>=0
    count=len(set(labels[keep]))
    silhouette=np.nan; db=np.nan
    if count>=2 and keep.sum()>count:
        silhouette=sampled_silhouette(x[keep],labels[keep])
        db=float(davies_bouldin_score(x[keep],labels[keep]))
    return dict(clusters=count,noise_fraction=float((~keep).mean()),
                silhouette=silhouette,davies_bouldin=db)


def fit_supervised(train):
    """All configurations fixed before holdout scoring; no random validation split."""
    fitted={}
    for energy in [False,True]:
        name='energy_calendar' if energy else 'calendar_only'
        x=feature_matrix(train,energy)
        for event in ['negative_price','high_price']:
            if train[event].nunique()!=2:
                raise ValueError(f'Training needs both classes for {event}')
            fitted[('random_forest',name,event)]=RandomForestClassifier(**RF_CONFIG).fit(x,train[event])
        fitted[('gradient_boosting',name,'SpotPriceEUR')]=HistGradientBoostingRegressor(**REG_CONFIG).fit(x,train.SpotPriceEUR)
    return fitted
