# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:09:39 2021
@author: V165809
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import mutual_info_score 
import networkx as nx
#import fastcluster
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster,linkage
from scipy.spatial.distance import squareform

def mpPDF(var,q,pts):
    ##Marcenko Pasteur pdf
    #=q=T/N
    eMin,eMax=var*(1-(1.0/q)**.5)**2,var*(1+(1.0/q)**.5)**2
    eVal=np.linspace(eMin,eMax,pts)
    pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf=pd.Series(pdf,index=eVal)
    return pdf


def getPCA(matrix):
    #Get eVal,eVec from hermitian matrix
    eVal,eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1] #arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec


def fitKDE(obs,bwidth=.25,kernel='gaussian',x=None):
    #from sklearn.neighbors.kde import KernelDensity 
    from sklearn.neighbors import KernelDensity
    #fit Kernel to a Series of observations and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape)==1:
        obs=obs.reshape(-1,1)
    kde=KernelDensity(kernel=kernel,bandwidth=bwidth).fit(obs)
    if x is None:
        x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:
        x=x.reshape(-1,1)
    logProb=kde.score_samples(x) #log-density
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf

def getRndCov(nCols,nFacts):
    w=np.random.normal(size=(nCols,nFacts))
    cov=np.dot(w,w.T) #random cov matirx not full rank
    cov+=np.diag(np.random.uniform(size=nCols)) #full rank cov
    return cov

def cov2corr(cov):
    std=np.sqrt(np.diag(cov))
    corr=cov/np.outer(std,std)
    corr[corr<-1],corr[corr>1]=-1,1
    return corr


def Errpdfs(var,eVal,q,bWidth,pts):
    #fit Error
    pdf0=mpPDF(var,q,pts)
    pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values)
    sse=np.sum((pdf1-pdf0)**2)
    return sse

def findMaxEval(eVal,q,bWidth):
    from scipy.optimize import minimize
    # Find max random eVal by fitting Marcenko's dist
    err2=lambda x: Errpdfs(x,eVal,q,bWidth,pts=1000)
    #out=minimize(lambda *x:Errpdfs(*x),x0=[.5],args=(eVal,q,bWidth),
    #             bounds=((1e-5,1-1e-5)))
    np.random.seed(69)
    n=len(eVal)
    out=minimize(err2,x0=pd.Series(.5*np.ones(n)),bounds=n*[(1e-5,1-1e-5)])
    if out['success']:
        var=out['x'][0]
    else:
        var=1
    eMax=var*(1+(1.0/q)**.5)**2
    return eMax,var

def denoisedCorr(eVal,eVec,nFacts):
    #Remove noise from corr by Fixing random eigenvlaues
    eVal_=np.diag(eVal).copy()
    eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal.shape[0]-nFacts)
    eVal_=np.diag(eVal_)
    corr1=np.dot(eVec,eVal_).dot(eVec.T)
    corr1=cov2corr(corr1)
    return corr1
    
def denoisedCorr2(eVal,eVec,nFacts,alpha=0):
    #Remove noise from corr through target shrinkage
    eValL,eVecL=eVal[:nFacts,:nFacts],eVec[:,:nFacts]
    eValR,eVecR=eVal[nFacts:,nFacts:],eVec[:,nFacts:]
    corr0=np.dot(eVecL,eValL).dot(eVecL.T)
    corr1=np.dot(eVecR,eValR).dot(eVecR.T)
    corr2=corr0+alpha*corr1+(1-alpha)*np.diag(np.diag(corr1))
    return corr2


def corr2cov(corr,std):
    return corr*np.outer(std,std)        

def deNoiseCov(cov0,q,bWidth):
    corr0=cov2corr(cov0)
    eVal0,eVec0=getPCA(corr0)
    eMax0,var0=findMaxEval(np.diag(eVal0),q,bWidth)
    nFacts0=eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1=denoisedCorr(eVal0,eVec0,nFacts0)
    return cov2corr(corr1,np.diag(cov0)**.5)

def deNoiseCov2(X):
    cov0=X.cov()
    T=len(X.index)
    N=len(X.columns)
    q=(1.0*T)/N
    b=.01
    return deNoiseCov(cov0,q,b)


def varInfo(x,y,bins,norm=False):
    #variation of information
    cXY=np.histogram2d(x,y,bins)[0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    hX=ss.entropy(np.histogram(x,bins)[0]) #Marginal
    hY=ss.entropy(np.histogram(y,bins)[0]) #Marginal
    vXY=hX+hY-2*iXY
    if norm:
        hXY=hX+hY-iXY
        vXY/=hXY
    return vXY


def numBins(nObs,corr=None):
    #Optimal Number of Bins for discretization
    if corr is None:
        z=(8+324*nObs+12*(36*nObs+729*nObs**2)**.5)**(1.0/3)
        b=round(z/6.0+2.0/(3*z)+1.0/3)
    else:
        b=round(2**-.5*(1+(1+24*nObs/(1.0-corr**2))**.5)**.5)
    return int(b)


def varInfo2(x,y,norm=False):
    #Variation of Informaiton
    bXY=numBins(x.shape[0],corr=np.corrcoef(x,y)[0,1])
    cXY=np.histogram2d(x,y,bXY)[0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    hX=ss.entropy(np.histogram(x.values,bXY)[0])
    hY=ss.entropy(np.histogram(y.values,bXY)[0])
    vXY=hX+hY-2*iXY
    if norm:
        hXY=hX+hY-iXY
        vXY/=hXY
    return vXY

def mutualInfo(x,y,norm=False):
    #mutual information
    bXY=numBins(x.shape[0],corr=np.corrcoef(x,y)[0,1])
    cXY=np.histogram2d(x,y,bXY)[0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    if norm:
        hX=ss.entropy(np.histogram(x,bXY)[0])
        hY=ss.entropy(np.histogram(y,bXY)[0])
        iXY/=min(hX,hY)
    return iXY


    
def compute_mst_stats(corr):
    dist = (1 - corr) / 2
    G = nx.from_numpy_matrix(dist) 
    mst = nx.minimum_spanning_tree(G)

    features = pd.Series()
    features['mst_avg_shortest'] = nx.average_shortest_path_length(mst)


    closeness_centrality = (pd
                            .Series(list(nx
                                         .closeness_centrality(mst)
                                         .values()))
                            .describe())
    for stat in closeness_centrality.index[1:]:
        features[f'mst_centrality_{stat}'] = closeness_centrality[stat]

    return features

def compute_intravar_clusters(model_corr, Z, nb_clusters=5):
    from scipy.cluster import hierarchy
    clustering_inds = hierarchy.fcluster(Z, nb_clusters,
                                         criterion='maxclust')
    clusters = {i: [] for i in range(min(clustering_inds),
                                     max(clustering_inds) + 1)}
    for i, v in enumerate(clustering_inds):
        clusters[v].append(i)

    total_var = 0
    for cluster in clusters:
        sub_corr = model_corr[clusters[cluster], :][:, clusters[cluster]]
        sa, sb = np.triu_indices(sub_corr.shape[0], k=1)
        mean_corr = sub_corr[sa, sb].mean()
        cluster_var = sum(
            [(sub_corr[i, j] - mean_corr)**2 for i in range(len(sub_corr))
             for j in range(i + 1, len(sub_corr))])
        total_var += cluster_var
    
    return total_var


def compute_features_from_correl(model_corr):
    import fastcluster
    from scipy.cluster.hierarchy import cophenet
    
    n = model_corr.shape[0]
    a, b = np.triu_indices(n, k=1)
    
    features = pd.Series()
    coefficients = model_corr[a, b].flatten()

    coeffs = pd.Series(coefficients)
    coeffs_stats = coeffs.describe()
    for stat in coeffs_stats.index[1:]:
        features[f'coeffs_{stat}'] = coeffs_stats[stat]
    features['coeffs_1%'] = coeffs.quantile(q=0.01)
    features['coeffs_99%'] = coeffs.quantile(q=0.99)
    features['coeffs_10%'] = coeffs.quantile(q=0.1)
    features['coeffs_90%'] = coeffs.quantile(q=0.9)
    # eigenvals
    eigenvals, eigenvecs = np.linalg.eig(model_corr)
    permutation = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[permutation]
    eigenvecs = eigenvecs[:, permutation]

    pf_vector = eigenvecs[:, np.argmax(eigenvals)]
    if len(pf_vector[pf_vector < 0]) > len(pf_vector[pf_vector > 0]):
        pf_vector = -pf_vector

    features['varex_eig1'] = float(eigenvals[0] / sum(eigenvals))
    features['varex_eig_top5'] = (float(sum(eigenvals[:5])) / 
                                  float(sum(eigenvals)))
    features['varex_eig_top8'] = (float(sum(eigenvals[:8])) / 
                                   float(sum(eigenvals)))
    # Marcenko-Pastur (RMT)
    T, N = 252, n
    MP_cutoff = (1 + np.sqrt(N / T))**2
    # variance explained by eigenvals outside of the MP distribution
    features['varex_eig_MP'] = (
        float(sum([e for e in eigenvals if e > MP_cutoff])) /
        float(sum(eigenvals)))
    
    # determinant
    features['determinant'] = np.prod(eigenvals)
    
    # condition number
    features['condition_number'] = abs(eigenvals[0]) / abs(eigenvals[-1])


    # stats of the first eigenvector entries
    pf_stats = pd.Series(pf_vector).describe()
    if pf_stats['mean'] < 1e-5:
        return None
    for stat in pf_stats.index[1:]:
        features[f'pf_{stat}'] = float(pf_stats[stat])


    # stats on the MST
    features = pd.concat([features, compute_mst_stats(model_corr)],
                         axis=0)

    # stats on the linkage
    dist = np.sqrt(2 * (1 - model_corr))
    for algo in ['ward', 'single', 'complete', 'average']:
        Z = fastcluster.linkage(dist[a, b], method=algo)
        features[f'coph_corr_{algo}'] = cophenet(Z, dist[a, b])[0]
        
        # stats on the clusters
    Z = fastcluster.linkage(dist[a, b], method='ward')
    features['cl_intravar_2'] = compute_intravar_clusters(
        model_corr, Z, nb_clusters=2)
    features['cl_intravar_5'] = compute_intravar_clusters(
        model_corr, Z, nb_clusters=5)
    features['cl_intravar_10'] = compute_intravar_clusters(
        model_corr, Z, nb_clusters=10)
    features['cl_intravar_2-5'] = (
        features['cl_intravar_2'] - features['cl_intravar_5'])
    features['cl_intravar_5-10'] = (
        features['cl_intravar_5'] - features['cl_intravar_10'])    
    

    return features.sort_index()



def feats_ts(df_r,N):
    '''
    get the features from the above function in rolling N-periods
    correlation
    '''
    corr=np.corrcoef(df_r.T)
    test=compute_features_from_correl(corr)
    list_feat=list(test.index)
    df_aux=pd.DataFrame(columns=list_feat,index=df_r.index[N:])
    for t in range(len(df_r)-N):
        corr_aux=np.corrcoef(df_r.iloc[t:t+N].T)
        df_aux.iloc[t]=compute_features_from_correl(corr_aux)
    return df_aux



def featImpMDI(fit,featNames):
    from sklearn import tree
    df0={i:tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0=pd.DataFrame.from_dict(df0,orient='index')
    df0.columns=featNames
    df0=df0.replace(0,np.nan) #because max_features=1
    imp=pd.concat({'mean':df0.mean(),'std':df0.std()*df0.shape[0]**-.5},axis=1)
    imp/=imp['mean'].sum()
    return imp



def seriation(Z, dim, cur_index):
    if cur_index < dim:
        return [cur_index]
    else:
        left = int(Z[cur_index - dim, 0])
        right = int(Z[cur_index - dim, 1])
        return seriation(Z, dim, left) + seriation(Z, dim, right)


def compute_serial_matrix(dist_mat,method='ward'):
    '''
    '''
   
    N=len(dist_mat)
    flat_dist_mat=squareform(dist_mat)
    res_linkage=linkage(flat_dist_mat,method=method)
    res_order=seriation(res_linkage,N,N+N-2)
    seriated_dist = np.zeros((N,N))
    a,b=np.triu_indices(N,k=1)
    seriated_dist[a,b]=dist_mat[[res_order[i] for i in a],[res_order[j] for j in b]]
    seriated_dist[b,a]=seriated_dist[a,b]
    return seriated_dist,res_order,res_linkage

def compute_HRP_w(covariances,res_order):
    weights = pd.Series(1,index=res_order)
    clustered_alphas=[res_order]
    while len(clustered_alphas)>0:
        clustered_alphas = [cluster[start:end] for cluster in clustered_alphas
                            for start, end in ((0, len(cluster) // 2),
                                               (len(cluster) // 2, len(cluster)))
                            if len(cluster) > 1]
        for subcluster in range(0,len(clustered_alphas),2):
            left_cluster = clustered_alphas[subcluster]
            right_cluster = clustered_alphas[subcluster+1]
            
            left_subcovar=covariances[left_cluster].loc[left_cluster]
            inv_diag=1.0/np.diag(left_subcovar.values)
            parity_w = inv_diag/np.sum(inv_diag)
            left_cluster_var=np.dot(parity_w,np.dot(left_subcovar,parity_w))
            
            right_subcovar = covariances[right_cluster].loc[right_cluster]
            inv_diag=1.0/np.diag(right_subcovar.values)
            parity_w = inv_diag/np.sum(inv_diag)
            right_cluster_var = np.dot(parity_w,np.dot(right_subcovar,parity_w))
            
            alloc_factor = 1-left_cluster_var/(left_cluster_var+right_cluster_var)
            
            weights[left_cluster] *=alloc_factor
            weights[right_cluster] *=1-alloc_factor
    return weights




def rolling_HRP(data,T,l_alloc,link_type):
    '''

    Parameters
    ----------
    X : return data
    T : window lenght
    link_type : linkage used single, ward etc
    l_alloc: date where you re-allocate

    Returns
    -------
    None.

    '''
    w_=pd.DataFrame(index=data.index[T:],columns=data.columns)
    for t in range(len(data.index)-T+1):
        cov_=data.iloc[t:t+T].cov()
        corr_=data.iloc[t:t+T].corr()
        dist_=np.sqrt((1-corr_)*.5)
        ordered_dist_mat, res_order, res_linkage=compute_serial_matrix(dist_.values, method=link_type)
        try:
            if w_.index[t] in l_alloc:
                w_.iloc[t]=compute_HRP_w(cov_,cov_.columns[res_order])[data.columns]
        except IndexError:
            print('te pasaste')
                
    return w_




def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))


def HERC_allocation(covar,Z, clusters):
    nb_clusters = len(clusters)
    dim=len(covar)
    assets_weights = np.array([1.] * len(covar))
    clusters_weights = np.array([1.] * nb_clusters)
    clusters_var = np.array([0.] * nb_clusters)    
    for id_cluster, cluster in clusters.items():
        cluster_covar = covar[cluster, :][:, cluster]
        inv_diag = 1 / np.diag(cluster_covar)
        assets_weights[cluster] = inv_diag / np.sum(inv_diag)        
    for id_cluster, cluster in clusters.items():
        weights = assets_weights[cluster]
        clusters_var[id_cluster - 1] = np.dot(
            weights, np.dot(covar[cluster, :][:, cluster], weights))        
    for merge in range(nb_clusters - 1):
        #print('id merge:', merge)
        left = int(Z[dim - 2 - merge, 0])
        right = int(Z[dim - 2 - merge, 1])
        left_cluster = seriation(Z, dim, left)
        right_cluster = seriation(Z, dim, right)

        #print(len(left_cluster),
        #      len(right_cluster))

        ids_left_cluster = []
        ids_right_cluster = []
        for id_cluster, cluster in clusters.items():
            if sorted(intersection(left_cluster, cluster)) == sorted(cluster):
                ids_left_cluster.append(id_cluster)
            if sorted(intersection(right_cluster, cluster)) == sorted(cluster):
                ids_right_cluster.append(id_cluster)

        ids_left_cluster = np.array(ids_left_cluster) - 1
        ids_right_cluster = np.array(ids_right_cluster) - 1
        #print(ids_left_cluster)
        #print(ids_right_cluster)
        #print()
        alpha = 0
        left_cluster_var = np.sum(clusters_var[ids_left_cluster])
        right_cluster_var = np.sum(clusters_var[ids_right_cluster])
        alpha = left_cluster_var / (left_cluster_var + right_cluster_var)

        clusters_weights[ids_left_cluster] = clusters_weights[
            ids_left_cluster] * alpha
        clusters_weights[ids_right_cluster] = clusters_weights[
            ids_right_cluster] * (1 - alpha)
    for id_cluster, cluster in clusters.items():
        assets_weights[cluster] = assets_weights[cluster] * clusters_weights[
            id_cluster - 1]        
    return assets_weights


'''
def get_HERC(df_X,N,nc):
    
#    Using N window-rolling correlations,
#    get the weights of the above function
    
    
    df_w=pd.DataFrame(index=df_X.index[N:],columns=df_X.columns)
    for t in range(len(df_X.index)-N):
        corr_mat=np.corrcoef(df_X.iloc[t:t+N].T)
        dist=((1-corr_mat)/2.)**.5
        dim=len(dist)
        tri_a,tri_b=np.triu_indices(dim,k=1)
        #Z=fastcluster.linkage(dist[tri_a,tri_b],method='ward')
        #permutation=hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, dist[tri_a, tri_b]))
        #ordered_corr=corr_mat[permutation,:][:,permutation]        
        clustering_inds=fcluster(Z,nc,criterion='maxclust')
        clusters = {i: [] for i in range(min(clustering_inds),max(clustering_inds) + 1)}
        for i, v in enumerate(clustering_inds):
            clusters[v].append(i)
        covar=corr2cov(corr_mat,df_X.iloc[t:t+N].std())
        df_w.iloc[t]=HERC_allocation(covar,Z,clusters)
    return df_w
        
'''  

def getIVP(cov,**kargs):
    # compute inverse variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp

def getClusterVar(cov,cItems):
    # compute variance per cluster
    cov_=cov.loc[cItems,cItems]
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

def getQuasiDiag(link):
    #sort clusterd items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # num original items
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2)
        df0=sortIx[sortIx>=numItems]
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] #item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx.append(df0) #item2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) #re-index
    return sortIx.tolist()

def getRecBipart(cov,sortIx):
    # Compute HRP single Alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] #initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)//2),(len(i)//2,len(i))) if len(i)>1]
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1= cItems[i+1] # Cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha #weight 1
            w[cItems1] *=1-alpha
    return w

def getHRP(cov,corr):
    # Cosntruct a Hierarchical Portfolio
    corr,cov=pd.DataFrame(corr),pd.DataFrame(cov)
    dist=np.sqrt(.5*(1.0-corr))
    link=linkage(dist,'single')
    sortIx=getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recovering labels
    hrp=getRecBipart(cov,sortIx)
    return hrp.sort_index()



            
        
        