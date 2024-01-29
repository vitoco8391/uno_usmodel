# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:13:35 2023

@author: VíctorValdenegroCabr
"""

import numpy as np
import pandas as pd
import BD
import seaborn as sns
import datetime as dt
import statsmodels.api as sm



sns.set_style('whitegrid')


l_tickers=("SPX Index","RTY Index","NDX Index","LF98TRUU Index",
           "LUACTRUU Index","FEDL01 Index","FDTR Index",
           "GB3 Govt","GT2 Govt","GT5 Govt","GT10 Govt","GT30 Govt")

sql1='SELECT * FROM tabla_maestra_javier WHERE Ticker in '+str(l_tickers)

df1=BD.descarga_query(sql1)

df1=pd.pivot_table(df1,columns=['Ticker','Feature'],index='Date',values='Value')
df1.fillna(method='ffill',inplace=True)


#### Graficando los Risk Premiums
df_graph1=pd.DataFrame(index=df1.index,columns=['EY-UST10','EY-USHY','EY-USCorpIG'])
df_graph1['EY-UST10']=100.0/df1['SPX Index']['BEST_PE_RATIO']-df1['GT10 Govt']['PX_LAST']
df_graph1['EY-USHY']=100.0/df1['SPX Index']['BEST_PE_RATIO']-df1['LF98TRUU Index']['YIELD_TO_WORST']
df_graph1['EY-USCorpIG']=100.0/df1['SPX Index']['BEST_PE_RATIO']-df1['LUACTRUU Index']['YIELD_TO_WORST']


#df_graph1.plot(grid=True,title='S&P Earning Yield Fwd versus YTW')

#### Graficando los Yields de ese gráfico

df_graph2=pd.DataFrame(index=df1.index,columns=['UST10','USHY_YTW','USCorpIG_YTW','SPX_EYF','SPX_EY'])
df_graph2['USHY_YTW']=df1['LF98TRUU Index']['YIELD_TO_WORST']
df_graph2['USCorpIG_YTW']=df1['LUACTRUU Index']['YIELD_TO_WORST']
df_graph2['UST10']=df1['GT10 Govt']['PX_LAST']
df_graph2['SPX_EYF']=100.0/df1['SPX Index']['BEST_PE_RATIO']
df_graph2['SPX_EY']=100.0/df1['SPX Index']['PE_RATIO']

#df_graph2.plot(grid=True,title='Yield Levels')



### Acá el grafico ql que siempre hago

df_analysis1=pd.DataFrame(index=df1.index,columns=['PE_FWD','EY-UST10','EY-USHY_YTW','UST10','FedFund Rates'])
df_analysis1['PE_FWD']=df1['SPX Index']['BEST_PE_RATIO']
df_analysis1['EY-UST10']=100.0/df1['SPX Index']['BEST_PE_RATIO']-df1['GT10 Govt']['PX_LAST']
df_analysis1['EY-USHY_YTW']=100.0/df1['SPX Index']['BEST_PE_RATIO']-df1['LF98TRUU Index']['YIELD_TO_WORST']
df_analysis1['FedFund Rates']=df1['FEDL01 Index']['PX_LAST']
df_analysis1['UST10']=df1['GT10 Govt']['PX_LAST']


df_analysis1.fillna(method='ffill',inplace=True,limit=3)
df_analysis1['Period']=""
df_analysis1['REC']='Regular Period'


for t in df_analysis1.index:
    if ( (t<dt.datetime(2001, 12, 1)) and (t>dt.datetime(2001,2,28))):
        df_analysis1['Period'].loc[t]='Dot-Com '
        df_analysis1['REC'].loc[t]='Recessionary Period'
    elif ((t>dt.datetime(2007,9,30)) and  (t<dt.datetime(2009,7,1))):
        df_analysis1['Period'].loc[t]='GFC'
        df_analysis1['REC'].loc[t]='Recessionary Period'
    elif ((t>dt.datetime(2020,1,30)) and  (t<dt.datetime(2020,5,1))):
        df_analysis1['Period'].loc[t]='COVID'
        df_analysis1['REC'].loc[t]='COVID'
    elif ((t>dt.datetime(1990,6,30)) and  (t<dt.datetime(1991,4,1))):
        df_analysis1['Period'].loc[t]='1990 Downturn'
        df_analysis1['REC'].loc[t]='Recessionary Period'
    elif t.year==2020:
        df_analysis1['Period'].loc[t]='2020 ex COVID'
        df_analysis1['REC'].loc[t]='2020 ex COVID'
    elif t.year==2021:
        df_analysis1['Period'].loc[t]='2021'
        df_analysis1['REC'].loc[t]='2021'
    elif t.year==2022:
        df_analysis1['Period'].loc[t]='2022'
        df_analysis1['REC'].loc[t]='2022'
    elif t.year==2023:
        df_analysis1['Period'].loc[t]='2023'
        df_analysis1['REC'].loc[t]='2023'
df_analysis1['Period'].iloc[-1]='Actual Value'        
df_analysis1['REC'].iloc[-1]='Actual Value' 

df_analysis1['d_REC']=1*(df_analysis1['REC']=='Recessionary Period')
#data[]

df_analysis1['d_REC*FedFunds']=df_analysis1['d_REC']*df_analysis1['FedFund Rates']

### Gráfico ql que siempre Hago
sns.lmplot(data=df_analysis1,y='PE_FWD',x='FedFund Rates',hue='REC')

l_ix=df_analysis1[df_analysis1.Period==''].index

l_ix2=list(df_analysis1[df_analysis1.REC=="Regular Period"].index)+list(df_analysis1[df_analysis1.REC=="Recessionary Period"].index)

x1=sm.add_constant(df_analysis1['UST10'])
model1=sm.OLS(df_analysis1.PE_FWD.loc[l_ix].astype(float),x1.loc[l_ix].astype(float),missing='drop').fit()

x2=sm.add_constant(df_analysis1['FedFund Rates'])
model2=sm.OLS(df_analysis1.PE_FWD.loc[l_ix].astype(float),x2.loc[l_ix].astype(float),missing='drop').fit()


x3=sm.add_constant(df_analysis1[['d_REC','FedFund Rates','d_REC*FedFunds']])
model3=sm.OLS(df_analysis1.PE_FWD.loc[l_ix2].astype(float),x3.loc[l_ix2].astype(float),missing='drop').fit()


df_analysis=pd.DataFrame(columns=['PU Fair']+list(np.linspace(100,250,16)),index=np.linspace(3.0,6.0,13))

df_analysis_rec=pd.DataFrame(columns=['PU Fair Rec']+list(np.linspace(100,250,16)),index=np.linspace(3.0,6.0,13))

Actual_Price=4550
PU_Fwd_actual=21
eps_actual=216

for r in df_analysis.index:
    df_analysis['PU Fair'].loc[r]=np.float(model3.predict([1.0,0,r,0]))
    for eps in df_analysis.columns[1:]:
        df_analysis[eps].loc[r]=df_analysis['PU Fair'].loc[r]*eps

for r in df_analysis_rec.index:
    df_analysis_rec['PU Fair Rec'].loc[r]=np.float(model3.predict([1.0,1,r,r]))
    for eps in df_analysis.columns[1:]:
        df_analysis_rec[eps].loc[r]=df_analysis_rec['PU Fair Rec'].loc[r]*eps

out=pd.ExcelWriter('./out_valsyrates.xlsx')
df_analysis.to_excel(out,sheet_name='analysis')
df_analysis_rec.to_excel(out,sheet_name='analysis_rec')
out.save()



df_eps=df1['SPX Index'][['TRAIL_12M_EPS','BEST_EPS']]
df_eps['EPS_1Y_Surprise']=(df_eps['TRAIL_12M_EPS']/df_eps['BEST_EPS'].shift(252))-1
df_eps['SPX_YOY']=df1['SPX Index']['TOT_RETURN_INDEX_GROSS_DVDS'].pct_change(252)
df_eps['USHY_YOY']=df1['LF98TRUU Index']['PX_LAST'].pct_change(252)
df_eps['EY-USHY_YTW']=100.0/df1['SPX Index']['BEST_PE_RATIO']-df1['LF98TRUU Index']['YIELD_TO_WORST']
df_eps['Fed Rate']=df1['FDTR Index']['PX_LAST']

df_eps.iloc[-20*252:].to_excel('sp_vs_hy.xlsx')

##(df_eps['TRAIL_12M_EPS']/df_eps['BEST_EPS'].shift(-252)).shift(252).iloc[-20*252:].plot(title='EPS 1Y Surprises')



##sm.GLS(df_analysis1['EY-USHY_YTW'].loc[l_ix2].astype(float),x3.loc[l_ix2].astype(float),missing='drop').fit().summary()



