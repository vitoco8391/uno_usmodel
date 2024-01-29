# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 17:57:15 2023

@author: VíctorValdenegroCabr
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import percentileofscore

def simm(u,v,VI):
    aux=np.dot(np.dot(u-v,VI),u-v)
    return -.5*aux

def info(u,v,VI):
    return np.dot(np.dot(u-v,VI),u-v)


def relevance(xi,xt,X):
    
    x_hat=np.mean(X)
    VI=np.linalg.inv(X.cov())
    
    #r_it=simm(xi,xt,VI)+.5*(info(xi,x_hat,VI)+info(xt,x_hat,VI))
    r_it=np.dot(np.dot(xi-x_hat,VI),xt-x_hat)
    return r_it

def all_relevances(xi,X):
    x_hat=np.mean(X)
    VI=np.linalg.inv(X.cov())
    
    ri=pd.Series(index=X.index)
    for t in X.index:
        xt=X.loc[t]
        ri.loc[t]=np.dot(np.dot(xi-x_hat,VI),xt-x_hat)
    return ri
    
def all_relevances2(xi,X,q):
    '''
    this one includes the weights for a Partial Sample Regression  taking r*
    as a quantile from the relevance of past observations
    '''
    x_hat=np.mean(X)
    VI=np.linalg.inv(X.cov())
    
    ri=all_relevances(xi,X)
    
    s_partial=(ri[ri>=ri.quantile(q)].std())**2
    s_full=(ri.std())**2
    n=len(ri[ri>=ri.quantile(q)])
    wi=ri*(ri>=ri.quantile(q))*(1.0/(n-1))*s_partial/s_full
    
    
    return ri,wi
    

def kkt_mahd(u,X,col):
    '''
    col: the one with the classifications
    '''
    d_params={}
    l_cats=X[col].unique()
    
    l_vars=list(X.columns)
    l_vars.remove(col)
    df_results=pd.DataFrame(index=l_cats,columns=['dist','likelihood','Probability'])
    df_dp=pd.DataFrame(columns=l_cats,index=l_vars)
    df_vi=pd.DataFrame(columns=l_cats,index=l_vars)
    S_x=np.std(X[l_vars])
    
    for c in l_cats:
        X_aux=X[X[col]==c]
        x_avg=X_aux[l_vars].mean()
        VI=pd.DataFrame(np.linalg.inv(X_aux[l_vars].cov()),l_vars,l_vars)
        
        d_params.update({c:[x_avg,X_aux[l_vars].cov(),VI]})
        
        
        d_=np.dot(np.dot(u-x_avg,VI),u-x_avg)
        #d_=VI.dot(u-x_avg).dot(u-x_avg)
        ll_=(np.linalg.det(2*np.pi*X_aux[l_vars].cov())**(-.5))*np.exp(-.5*np.float(d_))
        
        df_results['dist'].loc[c]=d_
        df_results['likelihood'].loc[c]=ll_
    df_results['Probability']=df_results['likelihood']/(1e-5+np.sum(df_results['likelihood']))
    
    for c in l_cats:
        df_dp[c]=0
        aux=d_params[c][2].dot(u-d_params[c][0])
        #print(aux)
        for c1 in l_cats:
            if c1!=c:
                aux1=d_params[c1][2].dot(u-d_params[c1][0])
                #print(df_results.loc[c1].Probability*(aux1-aux))
                df_dp[c]=df_dp[c]+df_results.loc[c1].Probability*(aux1-aux)
        df_dp[c]=df_dp[c]*df_results.loc[c].Probability
        df_vi[c]=df_dp[c]*S_x
        df_vi[c]=df_vi[c]/np.sum(np.abs(df_vi[c]))
    return df_results,df_vi,d_params

def calculate_percentiles(data):
    percentiles = [percentileofscore(data, x) for x in data]
    return percentiles


def kkt_mahd3(u,X,col):
    '''
    col: the one with the classifications
    this one takes the percentiles of the distances so its easier to detect outliers
    '''
    d_params={}
    l_cats=X[col].unique()
    
    l_vars=list(X.columns)
    l_vars.remove(col)
    df_results=pd.DataFrame(index=l_cats,columns=['dist','ll1','ll2','Probability'])
    df_dp=pd.DataFrame(columns=l_cats,index=l_vars)
    df_vi=pd.DataFrame(columns=l_cats,index=l_vars)
    S_x=np.std(X[l_vars])
    
    for c in l_cats:
        X_aux=X[X[col]==c]
        x_avg=X_aux[l_vars].mean()
        VI=pd.DataFrame(np.linalg.inv(X_aux[l_vars].cov()),l_vars,l_vars)
        
        d_params.update({c:[x_avg,X_aux[l_vars].cov(),VI]})
        
        
        d_=np.dot(np.dot(u-x_avg,VI),u-x_avg)
        #print(d_)
        #pd_=np.array(calculate_percentiles(d_))
        #d_=VI.dot(u-x_avg).dot(u-x_avg)
        #ll_=(np.linalg.det(2*np.pi*X_aux[l_vars].cov())**(-.5))*np.exp(-.5*np.float(pd_))
        ll_=(np.linalg.det(2*np.pi*X_aux[l_vars].cov())**(-.5))
        #print(ll_)
        df_results['ll1'].loc[c]=ll_
        
        df_results['dist'].loc[c]=d_
    
    pd_=df_results['dist'].rank(pct=True)*100
    print(pd_)
    
    df_results['ll2']=df_results['ll1']*np.exp(-.5*pd_.astype(float))
    #df_results['likelihood']=pd_
    df_results['Probability']=df_results['ll2']/(1e-5+np.sum(df_results['ll2']))
    df_results['Prob2']=np.exp(-.5*pd_)/(1e-15+np.sum(np.exp(-.5*pd_)))
    
    
    for c in l_cats:
        df_dp[c]=0
        aux=d_params[c][2].dot(u-d_params[c][0])
        #print(aux)
        for c1 in l_cats:
            if c1!=c:
                aux1=d_params[c1][2].dot(u-d_params[c1][0])
                #print(df_results.loc[c1].Probability*(aux1-aux))
                df_dp[c]=df_dp[c]+df_results.loc[c1].Probability*(aux1-aux)
        df_dp[c]=df_dp[c]*df_results.loc[c].Probability
        df_vi[c]=df_dp[c]*S_x
        df_vi[c]=df_vi[c]/(1e-15+np.sum(1e-15+np.abs(df_vi[c])))
    return df_results,df_vi,d_params

def kkt_mahd2(u,X,col):
    try:
       df_results,df_vi,d_params=kkt_mahd(u,X,col)
    except np.linalg.LinAlgError:
        df_results,df_vi,d_params=kkt_mahd2(u,X.iloc[:-2],col)
    return df_results,df_vi,d_params

def psr(Y,X,r_target):
    '''
    Y: targt outcome
    X: vars, observations
    r_target: use quintiles
    '''
    y_hat=pd.Series(index=Y.index)
    rev_obs=pd.DataFrame(index=X.index,columns=X.index)
    for i in X.index:
        for t in X.index:
            rev_obs[i].loc[t]=relevance(X.loc[i],X.loc[t],X)
    
    sig_full=pd.Series(index=X.index)
    sig_partial=pd.Series(index=X.index)
    n=pd.Series(index=X.index)
    for t in X.index:
        sig_full.loc[t]=rev_obs[t].std()
        sig_partial.loc[t]=rev_obs[t][rev_obs[t]>=rev_obs[t].quantile(r_target)].std()
        n.loc[t]=len(rev_obs[t][rev_obs[t]>=rev_obs[t].quantile(r_target)])
    return y_hat
    

















