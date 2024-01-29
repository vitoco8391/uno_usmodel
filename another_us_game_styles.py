# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:43:04 2023

@author: VíctorValdenegroCabr
"""

import BD
import numpy as np
import pandas as pd
import statsmodels.api as sm
import help_functs as hf
import datetime as dtt
import mldp_functs as mlf
import kkt_functs as kktf
from sklearn.covariance import LedoitWolf
import sys

#l_ustickers=("SPX Index","MXUS000V Index","MXUS000G Index","MXUSMVOL Index","MXUSMMT Index","MXUS00SV Index","MXUS00SG Index","MXUSQU Index",
#    "RTY Index","S5INFT Index","S5ENRS Index","S5HLTH Index","S5COND Index","S5UTIL Index","S5FINL Index","S5INDU Index","S5CONS Index",
#    "S5TELS Index","S5MATR Index")


l_ustickers=("SPX Index","MXUS000V Index","MXUS000G Index","MXUSQU Index","MXUSMVOL Index","MXUSMMT Index","RTY Index","MXUS00SV Index",
             "MXUS00SG Index","NDX Index","SPW Index","SPXGARPT Index","SPXHDUT Index","S5COND Index","S5CONS Index","S5ENRS Index","S5FINL Index","S5HLTH Index","S5INDU Index","S5INFT Index","S5MATR Index",
             "S5RLST Index","S5TELS Index","S5UTIL Index","LBUSTRUU Index","LF98TRUU Index","LUCCTRUU Index","LUMSTRUU Index","LUACTRUU Index",
             "LUATTRUU Index","LT01TRUU Index")



sql_qryus='SELECT * FROM tabla_maestra_javier WHERE Ticker in '+str(l_ustickers)

df_us=BD.descarga_query(sql_qryus)

l_feats=df_us.Feature.unique()

d_feats={}

for f in l_feats:
    d_feats.update({f:pd.pivot_table(df_us[df_us.Feature==f],values='Value',index='Date',columns='Ticker')})
    d_feats[f].fillna(method='ffill',inplace=True)
    
d_feats['Price_Sent']=d_feats['BEST_TARGET_PRICE']/d_feats['PX_LAST']-1
d_feats['Price_Sent'].fillna(method='ffill',inplace=True)

d_feats['DY-EY']=d_feats['GROSS_AGGTE_DVD_YLD']/100-1.0/d_feats['PE_RATIO']
d_feats['DY-EY'].fillna(method='ffill',inplace=True)

d_feats['Earning/Sales']=d_feats['PX_TO_SALES_RATIO']/d_feats['PE_RATIO']
d_feats['Earning/Sales'].fillna(method='ffill',inplace=True)

d_feats['Sales']=d_feats['PX_LAST']/d_feats['PX_TO_SALES_RATIO']
d_feats['Sales'].fillna(method='ffill',inplace=True)    

for dt in [22,63,126,252]:
    d_feats['FutRet_'+str(dt)+'d']=d_feats['TOT_RETURN_INDEX_GROSS_DVDS'].pct_change(dt).shift(-dt)
    for j in ["LBUSTRUU Index","LF98TRUU Index","LUACTRUU Index","LUATTRUU Index","LT01TRUU Index","LUCCTRUU Index","LUMSTRUU Index"]:
        d_feats['FutRet_'+str(dt)+'d'][j]=d_feats['PX_LAST'][j].pct_change(dt).shift(-dt)
        d_feats['FutRet_'+str(dt)+'d'][j].fillna(method='ffill',inplace=True,limit=10)
    
    
for dt in [22,63,252]:
    d_feats['Price_Sent chg_'+str(dt)]=d_feats['Price_Sent'].diff(dt)
    d_feats['PMOM_'+str(dt)]=d_feats['TOT_RETURN_INDEX_GROSS_DVDS'].pct_change(dt)
    d_feats['EPS Fwd chg_'+str(dt)]=d_feats['BEST_EPS'].pct_change(dt)
    
d_feats['MACD']=hf.MACD(d_feats['PX_LAST'],12,26,9)[0]-hf.MACD(d_feats['PX_LAST'],12,26,9)[1]
d_feats['RSI_22']=hf.rsi(d_feats['PX_LAST'],22)

l_v=['RETURN_ON_ASSET','RETURN_COM_EQY','PX_TO_SALES_RATIO','PX_TO_BOOK_RATIO','PE_RATIO','BEST_PE_RATIO','GROSS_AGGTE_DVD_YLD',
     'FREE_CASH_FLOW_YIELD','DY-EY','Earning/Sales','BEST_EPS','EV_TO_T12M_EBITDA','Sales']

for v in l_v:
    for T in [1,3,5,10]:
        df_aux=d_feats[v]
        d_feats.update({v+'_Z'+str(T):(df_aux-df_aux.rolling(T*252).mean())/(1e-15+df_aux.rolling(252*T).std())})
        
        
df_av=pd.DataFrame(index=list(d_feats.keys()),columns=list(l_ustickers))
for c in df_av.columns:
    for f in df_av.index:
        try:
            df_av[c].loc[f]=d_feats[f][c].dropna().index[0]
        except IndexError:
            df_av[c].loc[f]=np.nan
        except KeyError:
            df_av[c].loc[f]=np.nan

"""
l_prices=('VIX Index','USGGBE10 Index','CSI Barc Index','USSW10 BGN Curncy','USSW2 BGN Curncy',
          'USSW5 BGN Curncy','GT10 Govt','GT2 Govt','GT30 Govt','GB3 Govt','USGG5Y5Y Index',
          'GSUSFCI Index','INJCJC Index','INJCSP Index','BFCIUS Index','MRI CITI INDEX','GFSI Index',
          'GT5 Govt','MOODCBAA Index',
          'CSI BBB Index','MOVE Index','USGGBE05 Index','USGGBE02 Index','CESIUSD Index','MRIEM Index',
          'MRIST Index','BASPCAAA Index','JPEIGLSP Index','USFOSC1 Curncy','GVLQUSD Index','VXEEM Index',
          'VVIX Index','GTEUR10Y Govt','FDTR Index','FEDL01 Index','CPI YOY Index','CPI XYOY Index','IP YOY Index',
          'NAPMPMI Index','NAPMNMI Index','SBOITOTL Index','NFP TYOY Index','USURTOT Index','PRUSTOT Index',
          'LEI YOY Index','JOLTTOTL Index','M1 Index','BCOM Index','BCOMEN Index','BCOMIN Index','BCOMPR Index',
          'BCOMAG Index','DXY Index','MXEF0CX0 Index','FXJPEMCS Index','JPMVXYGL Index','JPMVG71M Index',
          'JPMVEM1M Index','JPY Curncy')
"""

l_prices=("BCOM Index","BCOMEN Index","CL1 Comdty","CO1 Comdty","QS1 Comdty","XB1 Comdty","NG1 Comdty",
       "FN1 Comdty","DOESTCRD Index","BCOMIN Index","BCOMPR Index",
       "LA1 Comdty","HG1 Comdty","GC1 Comdty","Silv Comdty","IRCNQ004 index",
       "RBT1 Comdty","MRSELSCU Index","LMY1 Comdty","BCOMAG Index","QW1 Comdty",
       "WEATCHEL Comdty","C 1 Comdty","LC1 Comdty","KCA Comdty","DXY Index","EUR Curncy","JPY Curncy",
       "CHF Curncy","MXEF0CX0 Index","FXJPEMCS Index",
       "VIX Index","USGGBE10 Index","CSI Barc Index","USSW10 BGN Curncy","USSW2 BGN Curncy",
       "USSW5 BGN Curncy","GT10 Govt","GT2 Govt","GT30 Govt","GB3 Govt","USGG5Y5Y Index",
       "GSUSFCI Index","INJCJC Index","INJCSP Index","BFCIUS Index","MRI CITI INDEX","GFSI Index",
       "CIISCSEP Index","CIISGREP Index","CIISEMEP Index","GT5 Govt","MOODCBAA Index",
       "CSI BBB Index","MOVE Index","USGGBE05 Index","USGGBE02 Index","CESIUSD Index","MRIEM Index",
       "MRIST Index","BASPCAAA Index","JPEIGLSP Index","JPMVXYGL Index","JPMVG71M Index","JPMVEM1M Index",
       "VVIX Index","FDTR Index","NAPMNMI Index","NAPMPMI Index","NAPMPRIC Index",
       "NAPMEMPL Index","NAPMNEWO Index","NAPMNPRC Index","NAPMNEMP Index","NAPMNNO Index","SPGSCI Index","USFOSC1 Curncy",
       "GVLQUSD Index","CPI YOY Index","CPI XYOY Index","IP YOY Index","SBOITOTL Index","NFP TYOY Index","USURTOT Index","PRUSTOT Index",
       "LEI YOY Index","JOLTTOTL Index","M1 Index","CONSSENT Index","CONSEXP Index","CFNAI Index","COI YOY Index","PIDSSYOY Index","MTIBYOY Index",
       "CPTICHNG Index","MGT2TB Index","JLGPUSPH Index","JLGPUSCH Index","JLGPUSPI Index","JLGPUSPG Index",
       "JLGPUSPL Index","JLGPUSPP Index","JLGPUSCI Index","JLGPUSCG Index","JLGPUSCL Index","JLGPUSCP","SAHMRLRT Index","EPUCNUSD Index",
       "SFFRNEWS Index"
       )

l_r=["BCOM Index","BCOMEN Index","CL1 Comdty","CO1 Comdty","QS1 Comdty","XB1 Comdty","NG1 Comdty",
       "FN1 Comdty","DOESTCRD Index","BCOMIN Index","BCOMPR Index",
       "LA1 Comdty","HG1 Comdty","GC1 Comdty","Silv Comdty","IRCNQ004 index",
       "RBT1 Comdty","MRSELSCU Index","LMY1 Comdty","BCOMAG Index","QW1 Comdty",
       "WEATCHEL Comdty","C 1 Comdty","LC1 Comdty","KCA Comdty","DXY Index","EUR Curncy","JPY Curncy",
       "CHF Curncy","MXEF0CX0 Index","FXJPEMCS Index"]

l_d=['VIX Index','USGGBE10 Index','CSI Barc Index','USSW10 BGN Curncy','USSW2 BGN Curncy',
          'USSW5 BGN Curncy','GT10 Govt','GT2 Govt','GT30 Govt','GB3 Govt','USGG5Y5Y Index',
          'GSUSFCI Index','INJCJC Index','INJCSP Index','BFCIUS Index','MRI CITI INDEX','GFSI Index',
          'CIISCSEP Index','CIISGREP Index','CIISEMEP Index','GT5 Govt','MOODCBAA Index',
          'CSI BBB Index','MOVE Index','USGGBE05 Index','USGGBE02 Index','CESIUSD Index','MRIEM Index',
          'MRIST Index','BASPCAAA Index','JPEIGLSP Index',
          'JPMVXYGL Index','JPMVG71M Index','JPMVEM1M Index','VVIX Index','FDTR Index','NAPMNMI Index',
          'NAPMPMI Index',"NAPMPRIC Index",
          "NAPMEMPL Index","NAPMNEWO Index","NAPMNPRC Index","NAPMNEMP Index","NAPMNNO Index","SPGSCI Index",
          "USFOSC1 Curncy","GVLQUSD Index",'SPX_EY-UST10','SPX_DY-UST10','UST10-UST2','UST10-UST3M','SPX_EY-USHYYTW',
          'SPX_EY-USCorpIGYTW',"JLGPUSPH Index","JLGPUSCH Index","JLGPUSPI Index","JLGPUSPG Index",
          "JLGPUSPL Index","JLGPUSPP Index","JLGPUSCI Index","JLGPUSCG Index","JLGPUSCL Index","SAHMRLRT Index","EPUCNUSD Index","SFFRNEWS Index"]

l_yoy=["JOLTTOTL Index","M1 Index","CONSSENT Index","CONSEXP Index","CFNAI Index","PIDSSYOY Index",
       "USURTOT Index","PRUSTOT Index","CPTICHNG Index",'US JobOpens/Underemployment',"MGT2TB Index"]

l_lag_monthly=['CPI YOY Index','CPI XYOY Index','IP YOY Index','NAPMPMI Index',
               'NAPMNMI Index','SBOITOTL Index','NFP TYOY Index','USURTOT Index',
               'PRUSTOT Index','LEI YOY Index','JOLTTOTL Index','M1 Index',"NAPMPRIC Index",
               "NAPMEMPL Index","NAPMNEWO Index","NAPMNPRC Index","NAPMNEMP Index","NAPMNNO Index","M1 Index",
               "CONSSENT Index","CONSEXP Index","CFNAI Index","COI YOY Index","PIDSSYOY Index",
               "MTIBYOY Index","CPTICHNG Index","MGT2TB Index"]


sql_qry2='SELECT * FROM tabla_maestra_javier WHERE Ticker in '+str(l_prices)+' AND Feature="PX_LAST"'
df_2=BD.descarga_query(sql_qry2)

data_2=pd.pivot_table(df_2,index='Date',columns='Ticker',values='Value')

for j in l_lag_monthly:
    data_2[j]=data_2[j].shift(22)

#data_2['Sahm Rule']=data_2['USURTOT Index'].dropna().rolling(3).mean()-data_2['USURTOT Index'].dropna().rolling(12).min()
data_2['LEI-COI']=data_2['LEI YOY Index']-data_2['COI YOY Index']


data_2['SPX_EY-UST10']=100.0/d_feats['PE_RATIO']['SPX Index']-data_2['GT10 Govt']
data_2['SPX_DY-UST10']=100.0/d_feats['GROSS_AGGTE_DVD_YLD']['SPX Index']-data_2['GT10 Govt']
data_2['UST10-UST2']=(data_2['GT10 Govt']-data_2['GT2 Govt'])
data_2['UST10-UST3M']=(data_2['GT10 Govt']-data_2['GB3 Govt'])
data_2['SPX_EY-USHYYTW']=100/d_feats['PE_RATIO']['SPX Index']-d_feats['YIELD_TO_WORST']['LF98TRUU Index']
data_2['SPX_EY-USCorpIGYTW']=100/d_feats['PE_RATIO']['SPX Index']-d_feats['YIELD_TO_WORST']['LUACTRUU Index']
data_2['SPX_MACD']=d_feats['MACD']['SPX Index']
data_2['SPX_RSI22']=d_feats['RSI_22']['SPX Index']

data_2['US JobOpens/Underemployment']=data_2['JOLTTOTL Index']/data_2['USURTOT Index']

###los que van en nivel
l_feats_ini=['LEI-COI','UST10-UST3M','UST10-UST2','SPX_DY-UST10','SPX_EY-UST10',
             "CONSSENT Index","CONSEXP Index","CFNAI Index","COI YOY Index","PIDSSYOY Index",
             "MTIBYOY Index","CPTICHNG Index",'NAPMNMI Index','NAPMPMI Index',"NAPMPRIC Index",
             "NAPMEMPL Index","NAPMNEWO Index","NAPMNPRC Index","NAPMNEMP Index","NAPMNNO Index",
             "USURTOT Index","PRUSTOT Index",'JPMVXYGL Index','JPMVG71M Index','JPMVEM1M Index',
             'VIX Index','USGGBE10 Index','CSI Barc Index','USSW10 BGN Curncy','USSW2 BGN Curncy',
             'USSW5 BGN Curncy','GT10 Govt','GT2 Govt','GT30 Govt','GB3 Govt','USGG5Y5Y Index',
             'SPX_EY-USHYYTW','SPX_EY-USCorpIGYTW',"MGT2TB Index","JLGPUSPH Index","JLGPUSCH Index","JLGPUSPI Index","JLGPUSPG Index",
             "JLGPUSPL Index","JLGPUSPP Index","JLGPUSCI Index","JLGPUSCG Index","JLGPUSCL Index","SAHMRLRT Index","EPUCNUSD Index","SFFRNEWS Index"
             ]


for j in l_yoy:
    data_2[j+'_YOY']=data_2[j].dropna().diff(12)
    l_feats_ini.append(j+'_YOY')

for v in l_v:
    for T in [3,5]:
        data_2['SPX_'+v+'_Z'+str(T)]=d_feats[v+'_Z'+str(T)]['SPX Index']
        l_feats_ini.append('SPX_'+v+'_Z'+str(T))

for dt in [63,252]:
    for c in l_r:
        data_2[c+' ret'+str(dt)]=data_2[c].pct_change(dt)
        l_feats_ini.append(c+' ret'+str(dt))
    for c in l_d:
        data_2[c+' diff'+str(dt)]=data_2[c].diff(dt)
        l_feats_ini.append(c+' diff'+str(dt))
    for k in ['PMOM_','EPS Fwd chg_']:
        data_2['SPX_'+k+str(dt)]=d_feats[k+str(dt)]['SPX Index']
        l_feats_ini.append('SPX_'+k+str(dt))


for j in ['LF98TRUU Index','LUACTRUU Index']:
    data_2[str(j)+'_OAS']=d_feats['INDEX_OAS_SWAP_BP'][j]
    l_feats_ini.append(str(j)+'_OAS')
    for dt in [63,252]:
        data_2[str(j)+'_OAS_d'+str(dt)]=d_feats['INDEX_OAS_SWAP_BP'][j].diff(dt)
        l_feats_ini.append(str(j)+'_OAS_d'+str(dt))
    for T in [3,5]:
        data_2[str(j)+'_OAS_Z'+str(T)]=(d_feats['INDEX_OAS_SWAP_BP'][j]-d_feats['INDEX_OAS_SWAP_BP'][j].rolling(252*T).mean())/d_feats['INDEX_OAS_SWAP_BP'][j].rolling(252*T).std()
        l_feats_ini.append(str(j)+'_OAS_Z'+str(T))

'''
for dt in [63,252]:
    for c in l_r:
        data_2[c+' diff'+str(dt)]=data_2[c].pct_change(dt)
        l_feats_ini.append(c+' diff'+str(dt))
    for c in l_d:
        data_2[c+' diff'+str(dt)]=data_2[c].diff(dt)
        l_feats_ini.append(c+' diff'+str(dt))
'''

for dt in [252,500,750]:
    for j in ['LF98TRUU','LUACTRUU','LUATTRUU','LT01TRUU']:
        data_2['SPX_Corr_'+j+'_'+str(dt)]=d_feats['TOT_RETURN_INDEX_GROSS_DVDS']['SPX Index'].pct_change(1).rolling(dt).corr(d_feats['PX_LAST'][j+' Index'].pct_change(1))
        l_feats_ini.append('SPX_Corr_'+j+'_'+str(dt))

"""-----data a construir----"""

data_2.fillna(method='ffill',inplace=True)

df_av2=pd.Series(index=l_feats_ini)
for c in l_feats_ini:
    try:
        df_av2[c]=data_2[c].dropna().index[0]
    except IndexError:
        print('ojo con la serie '+str(c))

df_av2.dropna(inplace=True)
#sys.exit('Check Data Availability')



l_feats_ini=list(df_av2[df_av2<dtt.datetime(2003,12,31)].index)
#l_feats_ini.remove('JLGPUSCI Index diff252')

l_us2=["MXUS000V Index","MXUS000G Index","MXUSMVOL Index","MXUSMMT Index","MXUS00SV Index","MXUS00SG Index","MXUSQU Index",
       "RTY Index","NDX Index","SPW Index","SPXGARPT Index","SPXHDUT Index","S5INFT Index","S5ENRS Index","S5HLTH Index","S5FINL Index","LBUSTRUU Index","LF98TRUU Index","LUACTRUU Index",
       "LUATTRUU Index","LT01TRUU Index"]
data_2=data_2[l_feats_ini]

l_y=[]

for j in l_us2:
    for c in [22,63,126,252]:
        data_2[j+'_rel_'+str(c)]=d_feats['FutRet_'+str(c)+'d'][j]-d_feats['FutRet_'+str(c)+'d']['SPX Index']
        data_2[j+'_rel_'+str(c)].fillna(method='ffill',inplace=True)
        l_y.append(j+'_rel_'+str(c))

l_m=[]
for j in range(len(data_2.index)-1):
    if (data_2.index[j]>=dtt.datetime(2003,12,31)) and(data_2.index[j].month!=data_2.index[j+1].month):
        l_m.append(data_2.index[j])

l_m.append(data_2.index[-1])

l_mtrain=l_m[:145]
l_mtest=l_m[144:]

#sys.exit('chequea los intervalos de train/test')

df_t=pd.DataFrame(columns=l_y,index=l_feats_ini)
df_p=pd.DataFrame(columns=l_y,index=l_feats_ini)
df_corr=pd.DataFrame(columns=l_y,index=l_feats_ini)
#df_mi=pd.DataFrame(columns=l_y,index=l_feats_ini)
#df_lasso=pd.DataFrame(columns=l_y,index=l_feats_ini)

data_2=data_2.loc[:,~data_2.T.duplicated(keep='first')]

for y_ in l_y:
    Y=data_2[y_].loc[l_mtrain]
    for x_ in l_feats_ini:
        X=sm.add_constant(data_2[x_].loc[l_mtrain])
        mod=sm.OLS(Y,X,missing='drop').fit()
        df_t[y_].loc[x_]=mod.tvalues[x_]
        df_p[y_].loc[x_]=mod.pvalues[x_]
        df_corr[y_].loc[x_]=Y.corr(data_2[x_].loc[l_mtrain])
        #df_mi[y_].loc[x_]=mlf.mutualInfo(Y,data_2[x_].loc[l_mtrain],False)

'''
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV,TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier


tscv=TimeSeriesSplit(gap=1,n_splits=10)
param_grid={'fit_intercept':[True,False],'positive':[True,False],'alpha':np.logspace(-3,3,1000)}

for y in l_y:
    Y=data_2[y_].loc[l_mtrain]
    lasso=Lasso()
    clf=RandomizedSearchCV(lasso,param_grid)
    clf.fit(y=Y,X=data_2[l_feats_ini].loc[l_mtrain])
    df_lasso[y]=clf.best_estimator_.coef_
'''
"""
p_tgt=.01
#d_vars={}
l_tot=[]
for y in l_y:
    #l_aux=list(df_mi[y][df_mi[y]>df_mi[y].mean()+df_mi[y].std()].index)
    l_aux=list(df_p[y][df_p[y]<p_tgt].index)
    '''
    if len(l_aux)>50:
        l_aux=list(df_p[y][df_p[y]<.25*p_tgt].index)
    
    ix1=data_2[y].loc[l_mtrain][data_2[y].loc[l_mtrain]>0].index
    ix2=data_2[y].loc[l_mtrain][data_2[y].loc[l_mtrain]<0].index
    
    mu1=data_2[l_aux].loc[ix1].mean()
    S1=data_2[l_aux].loc[ix1].cov()
    
    mu2=data_2[l_aux].loc[ix2].mean()
    S2=data_2[l_aux].loc[ix2].cov()
    
    d_vars.update({y:[l_aux,mu1,S1,mu2,S2]})
    '''
    l_tot=l_tot+l_aux

l_tot=list(set(l_tot))
"""


'''
d_p={}
for y in l_y:
    X_=data_2[d_vars[y][0]]
    mu_up=d_vars[y][1]
    S_up=d_vars[y][2]
    mu_down=d_vars[y][3]
    S_down=d_vars[y][4]
    if np.linalg.det(2*np.pi*S_up)<0:
        print('determinant up of '+str(y)+' negative' )
    elif np.linalg.det(2*np.pi*S_up)<1e-10:
        print('determinant up of '+str(y)+' near zero' )
    if np.linalg.det(2*np.pi*S_down)<0:
        print('determinant down of '+str(y)+' negative' )
    elif np.linalg.det(2*np.pi*S_down)<1e-10:
        print('determinant down of '+str(y)+' near zero' )
        
        
    d_p.update({y:kktf.p_rect(X_,mu_up,S_up,mu_down,S_down)})
'''


### Algoritmo de Boruta PY para seleccionar los features
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import LedoitWolf
from boruta import BorutaPy


d_selected={}
for y in l_y:
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)
    X_=np.array(data_2[l_feats_ini].loc[l_mtrain])
    y_=np.array(1*(data_2[y].loc[l_mtrain]>0))
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=10)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
    feat_selector.fit(X_, y_)
    
    asd=pd.DataFrame({'choosen':feat_selector.support_,'ranking':feat_selector.ranking_},index=l_feats_ini)
    d_selected.update({y:asd})
    
    
from collections import Counter
l_sel=[]
for y in l_y:
    l_sel=l_sel+list(d_selected[y]['choosen'][d_selected[y]['choosen']==True].index)


ser_sel=pd.Series(Counter(l_sel))
l_sel=list(ser_sel[ser_sel>=5].index)

data_3=data_2[l_sel]

mu_tot_train=data_3.loc[l_mtrain].mean()
s_tot_train=data_3.loc[l_mtrain].std()

data_Z3=(data_3-mu_tot_train)/s_tot_train
data_Z3.fillna(method='ffill',inplace=True)

#ei_val,ei_vec=np.linalg.eig(LedoitWolf().fit(data_Z3.loc[l_mtrain]).covariance_)
#ei_val,ei_vec=np.linalg.eigh(data_Z3.loc[l_mtrain].corr())
ei_val,ei_vec=mlf.getPCA(LedoitWolf().fit(data_Z3.loc[l_mtrain]).covariance_)
ei_val=np.diag(ei_val)
N=12
PCs=pd.DataFrame(ei_vec[:N].T,index=data_Z3.columns)
df_PC=pd.DataFrame(columns=PCs.columns, index=data_Z3.index)
for pc in df_PC.columns:
    df_PC[pc]=(PCs[pc].dot(data_Z3.T))
#sys.exit('Check Eigens')


df_y=pd.Series(index=df_PC.index)
for t in df_y.index:
    if t<=l_mtrain[-1]:
        df_y.loc[t]=l_mtrain[-1]
    elif t in l_mtest:
        df_y.loc[t]=t
df_y.fillna(method='ffill',inplace=True)


'''
d_varspc={}
for y in l_y:
    #l_aux=list(df_mi[y][df_mi[y]>df_mi[y].mean()+df_mi[y].std()].index)
    
    ix1=data_2[y].loc[l_mtrain][data_2[y].loc[l_mtrain]>0].index
    ix2=data_2[y].loc[l_mtrain][data_2[y].loc[l_mtrain]<0].index
    
    mu1=df_PC.loc[ix1].mean()
    #S1=df_PC.loc[ix1].cov()
    S1=pd.DataFrame(LedoitWolf().fit(df_PC.loc[ix1]).covariance_,df_PC.columns,df_PC.columns)
    print('Det up of '+str(y)+' '+str(np.linalg.det(S1)))
    
    mu2=df_PC.loc[ix2].mean()
    #S2=df_PC.loc[ix2].cov()
    S2=pd.DataFrame(LedoitWolf().fit(df_PC.loc[ix2]).covariance_,df_PC.columns,df_PC.columns)
    print('Det down of '+str(y)+' '+str(np.linalg.det(S2)))
    
    d_varspc.update({y:[mu1,S1,mu2,S2]})
'''

d_params={}
for y in l_y:
    print('Calculating params for '+str(y))
    d_={}
    ix1_train=data_2[y].loc[l_mtrain][data_2[y].loc[l_mtrain]>0].index
    ix2_train=data_2[y].loc[l_mtrain][data_2[y].loc[l_mtrain]<0].index
    mu1_train=df_PC.loc[ix1_train].mean()
    S1_train=df_PC.loc[ix1_train].cov()
    mu2_train=df_PC.loc[ix2_train].mean()
    S2_train=df_PC.loc[ix2_train].cov()
    for t in l_m:
        if t <= l_mtrain[-1]:
            d_.update({t:[mu1_train,S1_train,mu2_train,S2_train]})
        else:
            ix1=data_2[y].loc[l_m].loc[:t][data_2[y].loc[l_m].loc[:t]>0].index
            ix2=data_2[y].loc[l_m].loc[:t][data_2[y].loc[l_m].loc[:t]<0].index
            mu1=df_PC.loc[ix1].mean()
            S1=df_PC.loc[ix1].cov()
            mu2=df_PC.loc[ix2].mean()
            S2=df_PC.loc[ix2].cov()
            d_.update({t:[mu1,S1,mu2,S2]})
    d_params.update({y:d_})


d_p={}
d_vr={}

t_aux=df_PC.index[-1]
for y in l_y:
    '''
    mu_up=d_varspc[y][0]
    S_up=d_varspc[y][1]
    mu_down=d_varspc[y][2]
    S_down=d_varspc[y][3]
    '''
    '''
    if np.linalg.det(2*np.pi*S_up)<0:
        print('determinant up of '+str(y)+' negative' )
    elif np.linalg.det(2*np.pi*S_up)<1e-10:
        print('determinant up of '+str(y)+' near zero' )
    if np.linalg.det(2*np.pi*S_down)<0:
        print('determinant down of '+str(y)+' negative' )
    elif np.linalg.det(2*np.pi*S_down)<1e-10:
        print('determinant down of '+str(y)+' near zero' )
              '''
    #d_p.update({y:kktf.p_rect(df_PC,mu_up,S_up,mu_down,S_down)})
    print('Calculating Probabilities for '+str(y))
    mu_up=d_params[y][t_aux][0]
    S_up=d_params[y][t_aux][1]
    mu_down=d_params[y][t_aux][2]
    S_down=d_params[y][t_aux][3]

    d_vr.update({y:kktf.var_importance(df_PC,mu_up,S_up,mu_down,S_down)})  
    d_p.update({y:kktf.precvary(df_PC,df_y,d_params[y])})


df_probs=pd.DataFrame(columns=['Fut 1M','Fut 3M','Fut 6M','Fut 1Y'],index=l_us2)
for j in l_us2:
    df_probs['Fut 1M'].loc[j]=float(d_p[j+'_rel_22'].iloc[-3])
    df_probs['Fut 3M'].loc[j]=float(d_p[j+'_rel_63'].iloc[-3])
    df_probs['Fut 6M'].loc[j]=float(d_p[j+'_rel_126'].iloc[-3])
    df_probs['Fut 1Y'].loc[j]=float(d_p[j+'_rel_252'].iloc[-3])
df_probs['Avg.']=df_probs.T.mean()

df_probs.to_excel('./Probs_today.xlsx')


d_w={}
df_bt=pd.DataFrame(index=df_PC.loc[l_mtrain[0]:].index,columns=l_y)

for y in l_us2:
    print('Backtesting '+str(y))
    df_r=pd.DataFrame(index=d_p[y+'_rel_22'].loc[l_mtrain[0]:].index,columns=[y,'SPX Index'])
    df_r['SPX Index']=d_feats['TOT_RETURN_INDEX_GROSS_DVDS'][['SPX Index']].pct_change(1)
    if y in ['LBUSTRUU Index','LF98TRUU Index','LUATTRUU Index','LUACTRUU Index','LT01TRUU Index']:
        #df_r=d_feats['TOT_RETURN_INDEX_GROSS_DVDS'][['SPX Index']].pct_change(1).fillna(0)
        df_r[y]=d_feats['PX_LAST'][y].pct_change(1).fillna(0)
        #df_r=df_r[[y,'SPX Index']]
    else:
        #df_r=d_feats['TOT_RETURN_INDEX_GROSS_DVDS'][[y,'SPX Index']].pct_change(1).fillna(0)
        df_r[y]=d_feats['TOT_RETURN_INDEX_GROSS_DVDS'][[y]].pct_change(1)
    df_r.fillna(0,inplace=True)
    for c in [22,63,126,252]:
        df_w=pd.DataFrame(index=d_p[y+'_rel_'+str(c)].loc[l_mtrain[0]:].index,columns=df_r.columns)
        df_w[y]=d_p[y+'_rel_'+str(c)]['Prec'].loc[l_m]
        df_w['SPX Index']=1-df_w[y]
        df_w=df_w.shift(3)
        df_w.iloc[0]=0.5
        df_w=hf.weight_drifter(df_r.loc[l_mtrain[0]:],df_w.dropna())
        d_w.update({y+'_rel_'+str(c):df_w})
        df_bt[y+'_rel_'+str(c)]=hf.BT_P(df_w,(1+df_r.loc[l_mtrain[0]:]).cumprod(),y+'_rel_'+str(c))
    
    #lo mismo pero el promedio de las 3 probas
    df_w=pd.DataFrame(index=df_r.loc[l_mtrain[0]:].index,columns=df_r.columns)
    df_w[y]=(d_p[y+'_rel_22']+d_p[y+'_rel_63']+d_p[y+'_rel_126'])['Prec'].loc[l_m]/3
    df_w['SPX Index']=1-df_w[y]
    df_w=df_w.shift(3)
    df_w.iloc[0]=0.5
    df_w=hf.weight_drifter(df_r.loc[l_mtrain[0]:],df_w.dropna())
    d_w.update({y+'_rel_Avg':df_w})
    df_bt[y+'_rel_Avg']=hf.BT_P(df_w,(1+df_r.loc[l_mtrain[0]:]).cumprod(),y+'_rel_Avg')
    
    #Haciendo uno que compre 50/50
    df_w=pd.DataFrame(index=df_r.loc[l_mtrain[0]:].index,columns=df_r.columns)
    for t in l_m:
        df_w.loc[t]=.5
    df_w.iloc[0]=0.5
    df_w=hf.weight_drifter(df_r.loc[l_mtrain[0]:],df_w.dropna())
    df_bt['50/50_'+y+'_SPX']=hf.BT_P(df_w,(1+df_r.loc[l_mtrain[0]:]).cumprod(),'50/50_'+y+'_SPX')
    
    # Buy and Hold
    df_w=pd.DataFrame(index=df_r.loc[l_mtrain[0]:].index,columns=df_r.columns)
    for t in l_m:
        df_w[y].loc[t]=1
        df_w['SPX Index'].loc[t]=0
    df_w[y].iloc[0]=1
    df_w['SPX Index'].iloc[0]=0
    df_w=hf.weight_drifter(df_r.loc[l_mtrain[0]:],df_w.dropna())
    df_bt[y+'_Buy&Hold']=hf.BT_P(df_w,(1+df_r.loc[l_mtrain[0]:]).cumprod(),y+'_Buy&Hold')


d_yin={}
for t in range(len(l_mtrain)-12):
    if l_mtrain[t].month==12:
        d_yin.update({str(l_mtrain[t+12].year):(l_mtrain[t],l_mtrain[t+12])})


d_yout={}
for t in range(len(l_mtest)-12):
    if l_mtest[t].month==12:
        d_yout.update({str(l_mtest[t+12].year):(l_mtest[t],l_mtest[t+12])})
d_yout.update({'YTD':(l_mtest[-4],l_mtest[-1])})

d_yall={}
for t in range(len(l_m)-12):
    if l_m[t].month==12:
        d_yall.update({str(l_m[t+12].year):(l_m[t],l_m[t+12])})
d_yall.update({'YTD':(l_m[-4],l_m[-1])})

df_bt['SPX Index']=d_feats['TOT_RETURN_INDEX_GROSS_DVDS']['SPX Index']
df_bt=df_bt/df_bt.iloc[0]


stats_in=hf.expost_statizer2(df_bt.loc[l_mtrain[0]:l_mtrain[-1]].pct_change(1).fillna(0),pd.DataFrame(df_bt['SPX Index'].loc[l_mtrain[0]:l_mtrain[-1]].pct_change(1).fillna(0)),252,d_yin,[])
stats_out=hf.expost_statizer2(df_bt.loc[l_mtrain[-1]:].pct_change(1).fillna(0),pd.DataFrame(df_bt['SPX Index'].loc[l_mtrain[-1]:].pct_change(1).fillna(0)),252,d_yout,[])
stats_all=hf.expost_statizer2(df_bt.loc[l_mtrain[0]:].pct_change(1).fillna(0),pd.DataFrame(df_bt['SPX Index'].loc[l_mtrain[0]:].pct_change(1).fillna(0)),252,d_yall,[])

#sys.exit('out before explains')

d_explains={}
for y in l_y:
    d_explains.update({y:((d_vr[y]).dot(PCs.T)*data_Z3)})

for y in l_y:
    d_explains.update({y:(d_vr[y]).dot(PCs.T)*data_Z3})

df_bt.fillna(method='ffill',inplace=True)
out_kkt=pd.ExcelWriter('./results_kktspx.xlsx')
for j in l_us2:
    l_aux=[]
    for t in [22,63,126,252]:
        l_aux.append(str(j)+'_rel_'+str(t))
    l_aux.append(j+'_rel_Avg')
    l_aux.append('50/50_'+j+'_SPX')
    l_aux.append(j+'_Buy&Hold')
    l_aux.append('SPX Index')
    stats_all[l_aux].to_excel(out_kkt,sheet_name=j+'_statsall')
    stats_out[l_aux].to_excel(out_kkt,sheet_name=j+'_statsout')
    df_bt[l_aux].to_excel(out_kkt,sheet_name=j+'_BTP')

out_kkt.save()


asd=df_bt.iloc[-2]/df_bt.loc[l_mtrain[0]]-1-(df_bt['SPX Index'].iloc[-2]/df_bt['SPX Index'].loc[l_mtrain[0]]-1)
asd2=df_bt.iloc[-2]/df_bt.loc[l_mtrain[-1]]-1-(df_bt['SPX Index'].iloc[-2]/df_bt['SPX Index'].loc[l_mtrain[-1]]-1)


l_us_orderout=['SPX Index']
for j in l_us2:
    l_us_orderout.append(j+'_Buy&Hold')
    l_us_orderout.append('50/50_'+j+'_SPX')
    l_us_orderout.append(j+'_rel_Avg')
    for t in [22,63,126,252]:
        l_us_orderout.append(str(j)+'_rel_'+str(t))

df_btsumm=pd.DataFrame(index=l_us_orderout,columns=['Ann Ret All Sample','Ann Ret Train Sample','Ann Ret Out Sample'])
df_btsumm['Ann Ret All Sample']=stats_all[l_us_orderout].loc['Ann. Mean Return']
df_btsumm['Ann Ret Train Sample']=stats_in[l_us_orderout].loc['Ann. Mean Return']
df_btsumm['Ann Ret Out Sample']=stats_out[l_us_orderout].loc['Ann. Mean Return']

for t in [2016,2017,2018,2019,2020,2021,2022]:
    df_btsumm[str(t)]=stats_out[l_us_orderout].loc['Return on '+str(t)]

df_btsumm.to_excel('./btretsumm.xlsx')

df_explainsout=pd.DataFrame(columns=['Actual_Value']+l_y,index=l_sel)
df_explainsout['Actual_Value']=data_2[l_sel].iloc[-2]
for y in l_y:
    df_explainsout[y]=d_explains[y][l_sel].iloc[-2]

df_explainsout.to_excel('./model_explains.xlsx')






#### Doing a 2nd long only Backtest


l_us3=["MXUS000V Index","MXUS000G Index","MXUSMVOL Index","MXUSMMT Index",
       "MXUSQU Index","MXUS00SV Index","MXUS00SG Index"]


#df_w=pd.DataFrame(columns=l_us3,index=)
#df_p22=pd.DataFrame(columns=l_us3,index=df_w.index)
#df_p63=pd.DataFrame(columns=l_us3,index=df_w.index)
#df_p126=pd.DataFrame(columns=l_us3,index=df_w.index)

d_feats2={}
for dt in [22,63,126]:
    df_aux=pd.DataFrame(columns=l_us3,index=df_w.index)
    for j in l_us3:
        df_aux[j]=d_p[j+'_rel_'+str(dt)]['Prec']
    #df_p22[j]=d_p[j+'_rel_22']
    #df_p63[j]=d_p[j+'_rel_63']
    #df_p126[j]=d_p[j+'_rel_126']
    d_feats2.update({'Prob_'+str(dt):df_aux})

d_feats2.update({'Prob_avg':(d_feats2['Prob_22']+d_feats2['Prob_63']+d_feats2['Prob_126'])/3})

#df_pavg=(df_p22+df_p63+df_p126)/3


d_lo={}
d_so={}
d_ls={}

df_r=d_feats['TOT_RETURN_INDEX_GROSS_DVDS'][l_us3].pct_change(1)
for st in [22,63,126,'avg']:
    aux='Prob_'+str(st)
    df_w=pd.DataFrame(columns=l_us3,index=df_r.loc[l_mtrain[0]:].index)
    
    
    
### Outputs para el comite
data_2['Period']=np.nan
data_2['REC']=np.nan
for t in data_2.index:
    if ( (t<dtt.datetime(2001, 12, 1)) and (t>dtt.datetime(2001,2,28))):
        data_2['Period'].loc[t]='Dot-Com '
        data_2['REC'].loc[t]='Recessionary Period'
    elif ((t>dtt.datetime(2007,9,30)) and  (t<dtt.datetime(2009,7,1))):
        data_2['Period'].loc[t]='GFC'
        data_2['REC'].loc[t]='Recessionary Period'
    elif ((t>dtt.datetime(2020,1,30)) and  (t<dtt.datetime(2020,5,1))):
        data_2['Period'].loc[t]='COVID'
        data_2['REC'].loc[t]='COVID'
    elif ((t>dtt.datetime(1990,6,30)) and  (t<dtt.datetime(1991,4,1))):
        data_2['Period'].loc[t]='1990 Downturn'
        data_2['REC'].loc[t]='Recessionary Period'
    elif ((t>dtt.datetime(2009,6,30)) and  (t<dtt.datetime(2020,1,31))):
        data_2['Period'].loc[t]='post GFC pre COVID'
    elif ((t>dtt.datetime(2001,11,30)) and  (t<dtt.datetime(2007,9,28))):
        data_2['Period'].loc[t]='post .com pre GFC'
    elif t.year==2020:
        data_2['Period'].loc[t]='2020 ex COVID'
        data_2['REC'].loc[t]='2020 ex COVID'
    elif t.year==2021:
        data_2['Period'].loc[t]='2021'
        data_2['REC'].loc[t]='2021'
    elif t.year==2022:
        data_2['Period'].loc[t]='2022'
        data_2['REC'].loc[t]='2022'
    elif t.year==2023:
        data_2['Period'].loc[t]='2023'
        data_2['REC'].loc[t]='2023'
    else:
        data_2['Period'].loc[t]='Pre Dot-Com'
        
data_2['Period'].iloc[-1]='Actual Value'        
data_2['REC'].iloc[-1]='Actual Value' 



df_class=pd.read_excel('./class_vars_model_us.xlsx')
l_classs=df_class.Class_2.unique()
d_monitors={}

l_pers=data_2.Period.unique()
for j in l_classs:
    l1=list(df_class[df_class.Class_2==j].Ticker)
    df_aux=pd.DataFrame(index=l1,columns=l_pers)
    for T in l_pers:
        for i in l1:
            if i in list(data_2.columns):
                #z=(data_2[i].iloc[-1]-data_2[i][data_2.Period==T].mean())/data_2[i][data_2.Period==T].std()
                difff=data_2[i].iloc[-1]-data_2[i][data_2.Period==T].iloc[-1]
                df_aux[T].loc[i]=difff
                #df_aux[T].loc[i]=data_2[i][data_2.Period==T].iloc[-1]
                df_aux['Actual Value'].loc[i]=data_2[i].iloc[-1]
    d_monitors.update({j:df_aux})



#data['d_REC*FedFunds']=data['d_REC']*data['FedFund Rates']




