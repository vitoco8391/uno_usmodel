# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:02:53 2021

@author: VíctorValdenegroCabr
"""


import numpy as np
import pandas as pd



def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


def outlier_cleaner(df,Upq,Downq,method='last'):
    '''
    '''
    df_out=pd.DataFrame(index=df.index,columns=df.columns)
    for f in df.columns:
        df_aux=pd.DataFrame(index=df.index,columns=[f,'r','outlier','outlier_r','Clean_r','Value_cum','Clean_Value'])
        df_aux[f]=df[f]
        
        if method=='diff':
            df_aux['r']=df[f].diff(1)
        elif method=='pct':
            df_aux['r']=df[f].pct_change(1)
        elif method=='lndiff':
            df_aux['r']=np.log(df[f]).diff(1)
        else:
            print('choose a rela method you idiot')
        
        q_up=df[f].quantile(Upq)
        q_down=df[f].quantile(Downq)
        iqr=q_up-q_down
        out_low=q_down-1.5*iqr
        out_high=q_up+1.5*iqr
        
        df_aux['outlier']=1*((df[f]>out_high) |(df[f]<out_low)  )
        
        q_rup=df_aux.r.quantile(Upq)
        q_rdown=df_aux.r.quantile(Downq)
        iqrr=q_rup-q_rdown
        
        out_rlow=q_rdown-1.5*iqrr
        out_rhigh=q_rdown-1.5*iqrr
        df_aux['outlier_r']=1*((df_aux.r>out_rhigh) | (df_aux.r<out_rlow))
        
        ix_r= df_aux[df_aux['outlier_r']>0].index
        
        
        df_aux.Clean_r=df_aux.r
        df_aux.Clean_r.loc[ix_r]=0
        
        if method=='diff':
            df_aux.value_cum=df_aux.Clean_r.cumsum()+df_aux[f].dropna().iloc[0]
        elif method=='pct':
            df_aux.value_cum=(1+df_aux.Clean_r).cumprod()*df_aux[f].dropna().iloc[0]
        elif method=='lndiff':
            df_aux.value_cum=np.exp(np.log(df_aux.Clean_r).sum())*df_aux[f].dropna().iloc[0]
        
        
        ix_val=df_aux[df_aux['outlier']>0].index
        df_aux.Clean_Value=df_aux[f]
        df_aux.Clean_Value.loc[ix_val]=df_aux.Value_cum.loc[ix_val]
        
        df_out[f]=df_aux['Clean_Value']
    return df_out


def tsdf_toplot(df):
    '''
    
    Parameters
    ----------
    df : Time-series dataframe. It should have dates as index
    Returns
    -------
    None.
    '''
    l_aux=[]
    for t in df.index:
        for j in df.columns:
            if not df[j].isna().loc[t]:
                l_aux.append([t,j,df[j].loc[t]])
    return pd.DataFrame(l_aux, columns=['Date','Security','Value'])


def df_to_plot(df,x,y):
    '''
    x:name of x parameter (dataframe index)
    y:name of y parameter (dataframe columns)
    
    '''
    l=[]
    for y in df.columns:
        for j in df.index:
            l.append([j,y,df[y].loc[j]])
    return pd.DataFrame(l,columns=[x,y,'Value'])



def price2ret(df,h,x):
    '''
    Given a dataframe of prices df
    calculate the h-periods return
    x: continuos or discrete
    x='c' returns log prices
    x='d' returns discrete prices
    x=something else...DBAA
    '''
    r=np.log(df.astype(float)).diff(periods=h)
    if x=='d':
        return np.exp(r)-1
    elif x=='c':
        return r
    else:
        print('discrete or continuous, stupid')
        return('There is nothing here you idiot')
    

def ret2price(df,x):
    '''
    From an ordered dataframe of returns df
    return a dataframe of prices
    x=d-> transformation from discrete returns
    x=c-> transformation from continuos returns
    x=something else...DBAA
    '''
    
    if x in ['d','c']:
        P=pd.DataFrame(columns=df.columns, index=df.index)
        if x=='c':
            for t in range(len(P.index)):
                if t==0:
                    P.iloc[t]=1*np.exp(df.iloc[t])
                else:
                    P.iloc[t]=P.iloc[t-1]*np.exp(df.iloc[t])
        else:
            for t in range(len(P.index)):
                if t==0:
                    P.iloc[t]=1+df.iloc[t]
                else:
                    P.iloc[t]=P.iloc[t-1]*(1+df.iloc[t])
        return(P)
    else:
        print('discrete or continuous, stupid')
        return('There is nothing here you idiot')


def rolling_mean(df_r,N):
    '''
    df_r: dataframe of returns
    N: lookback period
    '''
    return df_r.rolling(N).mean()

def rolling_vol(df_r,N,T):
    '''
    df_r: dataframe of returns
    N: lookback period
    T: annualized factor
    '''
    return df_r.rolling(N).std()*np.sqrt(T)

def rolling_Pma(df_P,N):
    '''
    df_P: Prices DataFrame
    N: lookback window
    '''
    return df_P-df_P.rolling(N).mean()

def rolling_Pmadiff(df_P,N1,N2):
    '''
    df_P: Prices DataFrame
    N1: short lookback window
    N2: Long lookback window
    '''
    return df_P.rolling(N1).mean()-df_P.rolling(N2).mean()

def MACD(df_P,N1,N2,Ns):
    exp1=df_P.ewm(span=N1,adjust=False).mean()
    exp2=df_P.ewm(span=N2,adjust=False).mean()
    macd=exp1-exp2
    exp3=macd.ewm(span=Ns,adjust=False).mean()
    return macd,exp3

def PPO(df_P,N1,N2,Ns):
    exp1=df_P.ewm(span=N1,adjust=False).mean()
    exp2=df_P.ewm(span=N2,adjust=False).mean()
    ppo=(exp1-exp2)/exp2
    exp3=ppo.ewm(span=Ns,adjust=False).mean()
    return ppo,exp3

def DEMA(df_P,N):
    exp1=df_P.ewm(span=N,adjust=False).mean()
    exp2=exp1.ewm(span=N,adjust=False).mean()
    return 2*exp1-exp2

def TEMA(df_P,N):
    exp1=df_P.ewm(span=N,adjust=False).mean()
    exp2=exp1.ewm(span=N,adjust=False).mean()
    exp3=exp2.ewm(span=N,adjust=False).mean()
    return 3*(exp1-exp2)+exp3

def MACD_Baz(df_P,l_short,l_long):
    
    N=len(l_short)
    S=pd.Series(index=df_P.index)
    S.fillna(0,inplace=True)
    for k in range(N):
        exp1=df_P.ewm(span=l_short[k],adjust=False).mean()
        exp2=df_P.ewm(span=l_long[k],adjust=False).mean()
        xk=exp1-exp2
        yk=xk/df_P.rolling(63).std()
        zk=yk/yk.rolling(252).std()
        uk=zk*np.exp(-zk*zk/4)/.89
        S+=(1.0/N)*uk
    return S

def Cross_MACD_Baz(df_P,l_short,l_long):
    df_aux=pd.DataFrame(index=df_P.index,columns=df_P.columns)
    for i in df_P.columns:
        df_aux[i]=MACD_Baz(df_P[i],l_short,l_long)
    return df_aux
    

def iD_Mom(df_r,df_b,N):
    '''
    df_r:dataframe of returns
    df_b: enchmark's returns ....columns alpha and beta
    N: Lookback period
    '''
    #model.resid.mean()/model.resid.std()
    import statsmodels.api as sm
    df_aux=pd.DataFrame(columns=df_r.columns,index=df_r.index)
    for j in df_r.columns:
        for t in range(len(df_r.index[N:])):
            model=sm.OLS(df_r[j].iloc[t:t+N],df_b.iloc[t:t+N],missing='drop').fit()
            df_aux[j].iloc[t+N]=model.resid.sum()/(1e-15+model.resid.std())
    return df_aux

def Res_Vol(df_r,df_b,N):
    '''
    df_r:dataframe of returns
    df_b: enchmark's returns ....columns alpha and beta
    N: Lookback period
    '''
    #model.resid.mean()/model.resid.std()
    import statsmodels.api as sm
    df_aux=pd.DataFrame(columns=df_r.columns,index=df_r.index)
    for j in df_r.columns:
        for t in range(len(df_r.index[N:])):
            model=sm.OLS(df_r[j].iloc[t:t+N],df_b.iloc[t:t+N],missing='drop').fit()
            df_aux[j].iloc[t+N]=model.resid.std()
    return df_aux


def Res_Mom(df_r,df_b,N):
    '''
    df_r:dataframe of returns
    df_b: enchmark's returns ....columns alpha and beta
    N: Lookback period
    '''
    #model.resid.mean()/model.resid.std()
    import statsmodels.api as sm
    df_aux=pd.DataFrame(columns=df_r.columns,index=df_r.index)
    for j in df_r.columns:
        for t in range(len(df_r.index[N:])):
            model=sm.OLS(df_r[j].iloc[t:t+N],df_b.iloc[t:t+N],missing='drop').fit()
            df_aux[j].iloc[t+N]=model.resid.sum()
    return df_aux

def rolling_beta(df_r,df_b,N):
    '''
    df_r:dataframe of return
    df_b: benchamrk's returns' and 1...columns should be name Alpha and Beta
    N: lookback period
    '''
    #import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS
    df_aux=pd.DataFrame(columns=df_r.columns,index=df_r.index)
    for j in df_r.columns:
        df_aux[j]=RollingOLS(df_r[j],df_b,N).fit().params['Beta']
    return df_aux
    
def rolling_alpha(df_r,df_b,N):
    '''
    df_r:dataframe of return
    df_b: benchamrk's returns' and 1...columns should be name Alpha and Beta
    N: lookback period
    '''
    #import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS
    df_aux=pd.DataFrame(columns=df_r.columns,index=df_r.index)
    for j in df_r.columns:
        df_aux[j]=RollingOLS(df_r[j],df_b,N).fit().params['Alpha']
    return df_aux  


def rsi(price,N):
    ''' rsi indicator '''
    gain = (price-price.shift(1)).fillna(0) # calculate price gain with previous day, first row nan is filled with 0

    def rsiCalc(p):
        # subfunction for calculating rsi for one lookback period
        avgGain = p[p>0].sum()/N
        avgLoss = -p[p<0].sum()/N 
        rs = avgGain/avgLoss
        return 1.0 - 1.0/(1.0+rs+1e-15)
    # run for all periods with rolling_apply
    return gain.rolling(N).apply(rsiCalc) 


def stochRSI(price,N,T):
    aux=rsi(price,N)
    return (aux-aux.rolling(T).min())/(aux.rolling(T).max()-aux.rolling(T).min())


def LMom(price,lm,N):
    '''
    price: prices dataframe
    lm: list of month ends
    N: lookback period
    '''
    df_prices2=price.loc[lm]
    return price2ret(df_prices2,N,'d')

def Acc_Mom(price,lm,N1,N2):
    '''
    '''
    df_prices2=price.loc[lm]
    return LMom(1+LMom(df_prices2,lm,N1),lm,N2)


def LAdjMom(price,lm,N):
    '''
    price: prices dataframe
    lm: list of month ends
    N: lookback period
    '''
    if N==3:
        N2=60
    elif N==6:
        N2=126
    elif N==9:
        N2=190
    elif N==12:
        N2=252
    else:
        N2=21*N
    vol=rolling_vol(price2ret(price,1,'d'),N2,252)
    df_prices2=price.loc[lm]
    return price2ret(df_prices2,N,'d')/vol.loc[lm]

def P_MAdiff(price,lm,N1,N2):
    '''
    price: prices dataframe
    lm: list of month ends
    N1: lookback period short window
    N2: lookback period long window
    '''    
    return LMom(price,lm,N1)-LMom(price,lm,N2)

def PAdj_MAdiff(price,lm,N1,N2):
    '''
    price: prices dataframe
    lm: list of month ends
    N1: lookback period short window
    N2: lookback period long window
    '''    
    return LAdjMom(price,lm,N1)-LAdjMom(price,lm,N2)

def LAdjMom(price,lm,N):
    '''
    price: prices dataframe
    lm: list of month ends
    N: lookback period
    '''
    if N==3:
        N2=60
    elif N==6:
        N2=126
    elif N==9:
        N2=190
    elif N==12:
        N2=252
    else:
        N2=21*N
    vol=rolling_vol(price2ret(price,1,'d'),N2,252)
    df_prices2=price.loc[lm]
    return price2ret(df_prices2,N,'d')/vol.loc[lm]



def rank_mixer(d_feats,l,name):
    '''
    Parameters
    ----------
    d_feats : Dictionary with the factors to mix in Dataframe form
    l : list of factor names to mix
    name : output-name
    Returns
    -------
    None.
    '''
    from scipy.stats.mstats import winsorize
    df_aux=pd.DataFrame(index=d_feats[l[0]].index,columns=d_feats[l[0]].columns)
    df_aux.fillna(0,inplace=True)
    for j in l:
        df_Z=pd.DataFrame(index=d_feats[j].index,columns=d_feats[j].columns)
        for t in df_Z.index:
            df_Z.loc[t]=(d_feats[j].loc[t]-d_feats[j].loc[t].mean())/(1e-15+d_feats[j].loc[t].std())
            df_Z.loc[t]=winsorize(df_Z.loc[t], limits=[0.1, 0.1])
        df_aux=df_aux+df_Z/len(l)
    return df_aux

def rank_mixer2(d_feats,l,df_w,name):
    '''
    Parameters
    ----------
    d_feats : Dictionary with the factors to mix in Dataframe form
    l : list of factor names to mix
    df_w: Dataframe with weights in l
    name : output-name
    Returns
    -------
    None.
    '''
    from scipy.stats.mstats import winsorize
    df_aux=pd.DataFrame(index=d_feats[l[0]].index,columns=d_feats[l[0]].columns)
    df_aux.fillna(0,inplace=True)
    for j in l:
        df_Z=pd.DataFrame(index=d_feats[j].index,columns=d_feats[j].columns)
        for t in df_Z.index:
            df_Z.loc[t]=(d_feats[j].loc[t]-d_feats[j].loc[t].mean())/(1e-15+d_feats[j].loc[t].std())
            df_Z.loc[t]=winsorize(df_Z.loc[t], limits=[0.1, 0.1])
        df_aux=df_aux+df_Z*df_w[j].loc[t]
    return df_aux


def rank_mixer3(d_feats,l,l_signs,name):
    '''
    Parameters
    ----------
    d_feats : Dictionary with the factors to mix in Dataframe form
    l : list of factor names to mix
    df_w: Dataframe with weights in l
    name : output-name
    Returns
    -------
    None.
    '''
    from scipy.stats.mstats import winsorize
    df_aux=pd.DataFrame(index=d_feats[l[0]].index,columns=d_feats[l[0]].columns)
    df_aux.fillna(0,inplace=True)
    for j in range(len(l)):
        df_Z=pd.DataFrame(index=d_feats[l[j]].index,columns=d_feats[l[j]].columns)
        for t in df_Z.index:
            df_Z.loc[t]=(d_feats[l[j]].loc[t]-d_feats[l[j]].loc[t].mean())/(1e-15+d_feats[l[j]].loc[t].std())
            df_Z.loc[t]=winsorize(df_Z.loc[t], limits=[0.1, 0.1])
        df_aux=df_aux+df_Z*l_signs[j]
    return df_aux



def BT_P(d_w,d_P,Port_name):
    '''
    d_w: dataframe with the holdings
    d_P: dataframe with Prices
    It should return the portfolios Price series
    this function assumes we have Prices and weights already
    i.e. it doesn't do weight drifting
    '''
    r=price2ret(d_P,1,'d')
    df_X=pd.DataFrame(columns=[Port_name],index=d_w.index)
    for t in range(len(d_w.index)):
        if t==0:
            df_X.iloc[t]=0
        else:
            df_X.iloc[t]=np.sum(d_w.iloc[t-1]*r.loc[d_w.index[t]])
    return ret2price(df_X,'d')


def BT_PC(d_w,d_P,c,alloc,Port_name):
    '''
    d_w: dataframe with the holdings drifted
    d_P: dataframe with Prices
    alloc: list with the allocatioon dates
    c: transaction cost in basis points
    It should return the portfolios Price series
    this function assumes we have Prices a weights already
    i.e. it doesn't do weight drifting
    '''
    r=price2ret(d_P,1,'d')
    df_X=pd.DataFrame(columns=[Port_name],index=d_w.index)
    for t in range(len(d_w.index)):
        if t==0:
            df_X.iloc[t]=0
        elif d_w.index[t] in alloc:
            #dw=d_w.iloc[t-1]-d_w.iloc[t-2]*(1+r.loc[d_w.index[t-1]])/(1+np.dot(r.loc[d_w.index[t-1]],d_w.iloc[t-2]))
            df_X.iloc[t]=np.sum(d_w.iloc[t-1]*r.loc[d_w.index[t]])-c*np.abs(d_w.diff().iloc[t-1]).sum()
            #df_X.iloc[t]=np.sum(d_w.iloc[t-1]*r.loc[d_w.index[t]])-c*np.abs(d_w.diff().iloc[t-1]).sum()
        else:
            df_X.iloc[t]=np.sum(d_w.iloc[t-1]*r.loc[d_w.index[t]])
            
    return ret2price(df_X,'d')



def weight_drifter(d_r,d_w):
    '''
    d_w: weigth's dataframe at each rebalancing period
    d_r: returns dataframe, it has more periods than the weight's DataFrame (but including them)
    
    it should return a longer weight's dataframe'
    
    '''
    d_w2=pd.DataFrame(index=d_r.loc[d_w.index[0]:].index,columns=d_r.columns)
    
    for t in range(len(d_w2.index)):
        #print(d_w2.index[t])
        #print(d_w2.index[t] in list(d_w.index))
        if t==0: 
            d_w2.iloc[t]=d_w.loc[d_w2.index[t]]
        elif d_w2.index[t] in list(d_w.index):
            d_w2.iloc[t]=d_w.loc[d_w2.index[t]]
        else:
            R=np.sum(d_r.loc[d_w2.index[t]]*d_w2.iloc[t-1])
            d_w2.iloc[t]=d_w2.iloc[t-1]*(1+d_r.loc[d_w2.index[t]])/(1+R)
            d_w2.iloc[t]=d_w2.iloc[t]/(1e-15+np.sum(d_w2.iloc[t]))
    return d_w2

def expost_statizer2(d_rp,d_rb,T,dic_periods,l_alloc):
    '''
    Given a set of returns between some portfolios and benchmark
    return some basic stats
    this one doesnt get the turnover 
    '''
    
    #l=['Ann. Mean Return','Volatility','Sharpe Ratio','Skewness','Kurtosis','Prob Share Ratio','Ann. alpha','p-alpha','Beta','Information Ratio','ex-post T.E.','MaxDrawDown','Emp VaR 95%']
    l=['Ann. Mean Return','Volatility','Sharpe Ratio','Skewness','Kurtosis','Ann. alpha','p-alpha','Beta','Information Ratio','ex-post T.E.','MaxDrawDown','Emp VaR 95%','Turnover']
    for j in dic_periods.keys():
        l.append('Return on '+j)
    
    #l=['Ann. Mean Return','Volatility','Sharpe Ratio','Ann. alpha','p-alpha','Beta','Information Ratio','MaxDrawDown','Emp VaR 95%']
    df_stats=pd.DataFrame(columns=d_rp.columns,index=l)
    #from sklearn import linear_model
    import statsmodels.api as sm
    from scipy.stats import kurtosis, skew#, norm
    df_aux=d_rb.copy(deep=False)
    df_aux.columns=['Beta']
    df_aux['alpha']=1    
    for j in d_rp.columns:
        
        #df_stats[j].loc['Ann. Return']=(ret2price(pd.DataFrame(d_rp[j],columns=[j]),'d').iloc[-1])**(len(d_rp)/T)-1
        df_stats.at['Ann. Mean Return',j]=T*float(np.mean(d_rp[j]))
        
        
        df_stats.at['Volatility',j]=np.sqrt(T)*np.std(d_rp[j].astype(float),ddof=1)
        #print(np.sqrt(T)*np.std(d_rp[j].astype(float),ddof=1))
        #print(df_stats)
        SR=T*np.mean(d_rp[j].astype(float))/df_stats[j].loc['Volatility']
        df_stats.at['Sharpe Ratio',j]=float(SR)
        df_stats.at['Skewness',j]=skew(d_rp[j].astype(float))
        df_stats.at['Kurtosis',j]=kurtosis(d_rp[j].astype(float))        
        
        for i in dic_periods.keys():
            df_stats.at['Return on '+i,j]=float(ret2price(d_rp.loc[dic_periods[i][0]:dic_periods[i][1]],'d')[j].iloc[-1])-1
        
        
        mod=sm.OLS(d_rp[j].astype(float),df_aux.astype(float),dropna=True)
        df_stats.at['Ann. alpha',j]=(1+mod.fit().params['alpha'])**T-1
        df_stats.at['p-alpha',j]=mod.fit().pvalues['alpha']
        df_stats.at['Beta',j]=mod.fit().params['Beta']
        
        
        df_stats.at['ex-post T.E.',j]=np.sqrt(T)*np.std(d_rp[j].values.T-d_rb[d_rb.columns].values.T,ddof=1)
        df_stats.at['Information Ratio',j]=T*np.mean(d_rp[j].values.T-d_rb[d_rb.columns].values.T)
        df_stats.at['Emp VaR 95%',j]=d_rp[j].quantile(0.05)
        
        P_aux=ret2price(pd.DataFrame(d_rp[j]),'d')
        df_stats.at['MaxDrawDown',j]=float((P_aux/P_aux.cummax()-1).cummin().iloc[-1])
        
    return df_stats


def expost_statizer(d_rp,d_rb,T,dic_periods,d_w,l_alloc):
    '''
    Given a set of returns between some portfolios and benchmark
    return some basic stats
    '''
    
    #l=['Ann. Mean Return','Volatility','Sharpe Ratio','Skewness','Kurtosis','Prob Share Ratio','Ann. alpha','p-alpha','Beta','Information Ratio','ex-post T.E.','MaxDrawDown','Emp VaR 95%']
    l=['Ann. Mean Return','Volatility','Sharpe Ratio','Skewness','Kurtosis','Ann. alpha','p-alpha','Beta','Information Ratio','ex-post T.E.','MaxDrawDown','Emp VaR 95%','Turnover']
    for j in dic_periods.keys():
        l.append('Return on '+j)
    
    #l=['Ann. Mean Return','Volatility','Sharpe Ratio','Ann. alpha','p-alpha','Beta','Information Ratio','MaxDrawDown','Emp VaR 95%']
    df_stats=pd.DataFrame(columns=d_rp.columns,index=l)
    #from sklearn import linear_model
    import statsmodels.api as sm
    from scipy.stats import kurtosis, skew#, norm
    df_aux=d_rb.copy(deep=False)
    df_aux.columns=['Beta']
    df_aux['alpha']=1    
    for j in d_rp.columns:
        
        #df_stats[j].loc['Ann. Return']=(ret2price(pd.DataFrame(d_rp[j],columns=[j]),'d').iloc[-1])**(len(d_rp)/T)-1
        df_stats.at['Ann. Mean Return',j]=T*float(np.mean(d_rp[j]))
        
        
        df_stats.at['Volatility',j]=np.sqrt(T)*np.std(d_rp[j].astype(float),ddof=1)
        #print(np.sqrt(T)*np.std(d_rp[j].astype(float),ddof=1))
        #print(df_stats)
        SR=T*np.mean(d_rp[j].astype(float))/df_stats[j].loc['Volatility']
        df_stats.at['Sharpe Ratio',j]=float(SR)
        df_stats.at['Skewness',j]=skew(d_rp[j].astype(float))
        df_stats.at['Kurtosis',j]=kurtosis(d_rp[j].astype(float))        
        df_stats.at['Turnover',j]=(d_w[j].diff().loc[l_alloc]>0).T.sum().mean()/12
        #Terms for Probabilstic Sharpe Ratio
        #terms are returned back to daily basis
        #print(df_stats.loc['Sharpe Ratio'][j])
        
        #SRb=float((np.mean(d_rb.values.T))/np.std(d_rb.astype(float),ddof=1))
        #print(SR)
        #print(SRb)
        
        #aux=np.sqrt(len(d_rp)-1)*(SR-SRb)/np.sqrt(1+0.5*SR**2-df_stats[j].loc['Skewness']*SR+0.25*(df_stats[j].loc['Kurtosis']-3)*SR**2)
        #df_stats.at['Prob Share Ratio',j]=float(norm.cdf(aux))
        
        #mod_aux=linear_model.LinearRegression()
        #mod_aux.fit(d_rb.astype(float),d_rp[j].astype(float))
        for i in dic_periods.keys():
            df_stats.at['Return on '+i,j]=float(ret2price(d_rp.loc[dic_periods[i][0]:dic_periods[i][1]],'d')[j].iloc[-1])-1
        
        
        #df_stats[j].loc['Ann. alpha']=(1+float(mod_aux.intercept_))**T-1
        #df_stats[j].loc['Beta']=float(mod_aux.coef_)     
        #print(df_stats)
        mod=sm.OLS(d_rp[j].astype(float),df_aux.astype(float),dropna=True)
        df_stats.at['Ann. alpha',j]=(1+mod.fit().params['alpha'])**T-1
        df_stats.at['p-alpha',j]=mod.fit().pvalues['alpha']
        df_stats.at['Beta',j]=mod.fit().params['Beta']
        
        
        df_stats.at['ex-post T.E.',j]=np.sqrt(T)*np.std(d_rp[j].values.T-d_rb[d_rb.columns].values.T,ddof=1)
        df_stats.at['Information Ratio',j]=T*np.mean(d_rp[j].values.T-d_rb[d_rb.columns].values.T)
        df_stats.at['Emp VaR 95%',j]=d_rp[j].quantile(0.05)
        
        P_aux=ret2price(pd.DataFrame(d_rp[j]),'d')
        df_stats.at['MaxDrawDown',j]=float((P_aux/P_aux.cummax()-1).cummin().iloc[-1])
        
    return df_stats



def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int)
    sortIx=pd.Series([link[-1,0],link[-1,1]])
    numItems=link[-1,3] # number of original items
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
        df0=sortIx[sortIx>=numItems] # find clusters
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1)
        sortIx=sortIx.append(df0) # item 2
        sortIx=sortIx.sort_index() # re-sort
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()


def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov)
    ivp/=ivp.sum()
    return ivp

def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,int(len(i)/2)), (int(len(i)/2),len(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0)
            cVar1=getClusterVar(cov,cItems1)
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2
    return w

def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist=((1-corr)/2.)**.5 # distance matrix
    return dist


def getHRP(cov,corr):
    # Construct a hierarchical portfolio
    import scipy.cluster.hierarchy as sch
    corr,cov=pd.DataFrame(corr),pd.DataFrame(cov)
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recover labels
    hrp=getRecBipart(cov,sortIx)
    return hrp.sort_index()


def cov2corr(cov):
    std=np.sqrt(np.diag(cov))
    corr=cov/np.outer(std,std)
    corr[corr<-1],corr[corr>1]=-1,1
    return corr

def get_HRP(df_ls,N):
    #calls the above funtion
    from sklearn.covariance import LedoitWolf
    r=price2ret(df_ls,1,'d')
    r.fillna(0)
    df_aux=pd.DataFrame(columns=df_ls.columns,index=r.index[N:])
    for t in range(len(r.index)-N):
        cov=LedoitWolf().fit(r.iloc[t:t+N].astype(float).fillna(0)).covariance_
        corr=cov2corr(cov)
        df_aux.iloc[t]=getHRP(cov,corr).values
    return df_aux


def Factor_Mom(df_ls,lm,N1,N):
    #Long top N1 top and short N1 bottom in N rolling window
    from scipy.stats import rankdata
    df_mom=LMom(df_ls,lm,N)
    df_aux=pd.DataFrame(columns=df_mom.columns,index=df_mom.index)
    for j in df_aux.index:
        rank=rankdata(df_mom.loc[j],method='dense')
        Nl=(rank<=N1).sum()
        Ns=(rank>=N1+1).sum()
        for i in range(len(df_aux.columns)):
            if rank[i]<=N1:
                df_aux[df_aux.columns[i]].loc[j]=1.0/Nl
            elif rank[i]>=N1+1:
                df_aux[df_aux.columns[i]].loc[j]=-1.0/Ns
    return df_aux
        

def Factors_to_portfolio(df_wfacts,d_wx,l_m):
    ##returns the 
    ##d_wx en el formato ya definido
    df_aux=pd.DataFrame(index=l_m,columns=d_wx[df_wfacts.columns[0]]['Long-Short w'].columns)
    df_aux.fillna(0,inplace=True)
    N=1.0/len(df_wfacts)
    for j in l_m:
        for i in df_wfacts.columns:            
            df_aux.loc[j]=df_aux.loc[j]+df_wfacts[i].loc[j]*d_wx[i]['Long-Short w'].loc[j]
            df_aux.loc[j][df_aux.loc[j]>0]=df_aux.loc[j][df_aux.loc[j]>0]/np.sum(df_aux.loc[j][df_aux.loc[j]>0])
            df_aux.loc[j][df_aux.loc[j]<0]=df_aux.loc[j][df_aux.loc[j]<0]/np.sum(-df_aux.loc[j][df_aux.loc[j]<0])
    
    return df_aux


def Z_scorizer(df):
    df_Z=pd.DataFrame(index=df.index,columns=df.columns)
    for t in df.index:
        aux=(df.loc[t]-df.loc[t].mean())/df.loc[t].std(ddof=1)
        df_Z.loc[t]=aux
    return df_Z

def Naive_RP(df_P,N,T):
    
    df_r=price2ret(df_P,1,'d')
    vols=rolling_vol(df_r,N,T)
    vols2=1.0/(1e-15+vols*2)
    df_aux=pd.DataFrame(index=vols2.index,columns=vols2.columns)
    for t in df_aux.index:
        df_aux.loc[t]=vols2.loc[t]/(1e-15+np.sum(vols2.loc[t]))
    return df_aux
    



def PCA_rec(X_train,nComp):
    '''
    
    Parameters
    ----------
    X_train : DataFrame with the original data, in returns format
    nComp : number of components to use
    Returns
    -------
    Re-constructed returns
    '''
    import sklearn.decomposition
    pca = sklearn.decomposition.PCA(n_components=nComp,random_state=42)
    pca.fit(X_train)
    x_train_pca=pca.fit_transform(X_train)
    x_projected=pca.inverse_transform(x_train_pca)
    return pd.DataFrame(x_projected,index=X_train.index,columns=X_train.columns)



def recursive_res(X,N,nComp):
    '''
    
    Parameters
    ----------
    X_train : DataFrame with the original data, in returns format
    N: rolling window length
    nComp : number of components to use
    
    Returns
    -------
    Re-constructed returns
    '''
    
    df_aux=pd.DataFrame(columns=X.columns,index=X.index[N:])
    for t in range(len(X.index[:-N])):
        aux=PCA_rec(X.iloc[t:t+N],nComp)
        #res=(1+X.iloc[t:t+N]).cumprod()-(1+aux).cumprod()
        res=(1+X.iloc[t:t+N]-aux).cumprod()
        res=(1+X.iloc[t:t+N]).cumprod()/((1+aux).cumprod())-1
        #df_aux.iloc[t]=res.iloc[-1]
        df_aux.iloc[t]=res.iloc[-1]
    return df_aux


    
    
def single_models_to_weight(d_model,d_X,lev,w_bench,l_train,l_test,types):
    '''
    this function takes the model a for and uses its views to
    generate to predict, generate the rankings and 
    and generate the weighting scheme in and out of sample
    type_: list where 
    '''
    from scipy.stats import rankdata
    cols=list(d_model.keys())
    N=len(cols)
    
    df_pred_train=pd.DataFrame(index=l_train,columns=cols)
    df_pred_test=pd.DataFrame(index=l_test,columns=cols)
    df_rank_train=pd.DataFrame(index=l_train,columns=cols)
    df_rank_test=pd.DataFrame(index=l_test,columns=cols)
    
    w_train=pd.DataFrame(index=l_train,columns=cols)
    w_test=pd.DataFrame(index=l_test,columns=cols)
    for j in cols:
        if types[0]=='Regression':
            df_pred_train[j]=d_model[j].predict(d_X[j].loc[l_train])
            df_pred_test[j]=d_model[j].predict(d_X[j].loc[l_test])
    for t in l_train:
        df_rank_train.loc[t]=N+1-rankdata(df_pred_train.loc[t])
        if types[1]=='EW':
            temp=df_rank_train.loc[t]
            w_train.loc[t]=w_bench.loc[t]-lev*np.sign(temp-np.median(temp))
    for t in l_test:
        df_rank_test.loc[t]=N+1-rankdata(df_pred_test.loc[t])
        if types[1]=='EW':
            temp=df_rank_test.loc[t]
            w_test.loc[t]=w_bench.loc[t]-lev*np.sign(temp-np.median(temp))
            
    d_aux={'w_train':w_train,'w_test':w_test,'pred_train':df_pred_train,'pred_test':df_pred_test}
    return d_aux


def model_mixer(d_models,w_bench,l_train,l_test,list_models):
    '''
    '''
    
    
    

def tuned_setup(which_model):
    
    
    if which_model=='RFR':
        from sklearn.ensemble import RandomForestRegressor
        model=RandomForestRegressor()
        tuned_grid = {'bootstrap': [True, False],
              'random_state':[42],
               'max_depth': [2,5,10,20,30,100, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': list(set(map(int,np.round(np.logspace(0,3,30),0))))}
        
    elif which_model=='XGBoostR':
        import xgboost
        model=xgboost.XGBRegressor()
        tuned_grid={'n_estimators': list(set(map(int,np.round(np.logspace(0,3,30),0)))),
           'learning_rate':np.logspace(-4,2,50),
           'max_depth': [3, 4, 5, 6, 7, 8, 9],
           'min_child_weight': [1, 2, 3],
           'subsample':np.linspace(0.1,0.9),
           'colsample_bytree': [0.6, 0.8, 1.0],
           }
    
    elif which_model=='SVR':
        from sklearn.svm import SVR
        model=SVR()
        tuned_grid={'kernel':['linear','sigmoid','rbf'],'C':np.logspace(-4,2,50),'gamma':['auto','scale'],
                    'epsilon':np.logspace(-3,-1,20)}
    
    elif which_model=='Lasso':
        from sklearn.linear_model import Lasso
        model=Lasso()
        tuned_grid={'alpha':np.logspace(-3,1,30),'fit_intercept':[True,False],'positive':[True,False]}

    elif which_model=='AdaBoostR':
        from sklearn.ensemble import AdaBoostRegressor
        model=AdaBoostRegressor()
        tuned_grid={'n_estimators': list(set(map(int,np.round(np.logspace(0,3,30),0)))),
                    'learning_rate':np.logspace(-4,2,50),
                    'random_state':[42],
                    'loss':['linear','square','exponential']}
    
    elif which_model=='KNNR':
        from sklearn.neighbors import KNeighborsRegressor
        model=KNeighborsRegressor()
        tuned_grid={'n_neighbors':np.linspace(2,30,29,dtype=int),'weights':['uniform','distance'],
                'algorithm':['auto','balltree','kd_tree','brute'],'leaf_size':np.linspace(2,30,29,dtype=int)
            }
    elif which_model=='RadNNR':
        from sklearn.neighbors import RadiusNeighborsRegressor
        model=RadiusNeighborsRegressor()
        tuned_grid={'radius':np.logspace(-3,3,50),'weights':['uniform','distance'],
                'algorithm':['auto','balltree','kd_tree','brute'],'leaf_size':np.linspace(2,30,29,dtype=int)
            }
        
     
    return {'model':model,'tuned':tuned_grid}
        
