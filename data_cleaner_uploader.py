# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:47:02 2022

@author: VíctorValdenegroCabr
"""

import numpy as np
import pandas as pd
import datetime
from sqlalchemy import create_engine
import sys





data_bulk=pd.read_csv('./your_data.txt')

    

epoch0=datetime.datetime(1899,12,31)


l_dates=data_bulk.Date.unique()

'''
rep={}

for j in l_dates:
    print(j)
    l_aux=data_bulk[data_bulk.Date==j].index
    if '/' in j:
        #data_bulk.Date.loc[l_aux]=datetime.datetime.strptime(j,'%d/%m/%Y')
        rep.update({j:datetime.datetime.strptime(j,'%d/%m/%Y')})
        #data_bulk.replace(rep)
    elif '-' in j:
        #data_bulk.Date.loc[l_aux]=datetime.datetime.strptime(j,'%d-%m-%Y')
        rep.update({j:datetime.datetime.strptime(j,'%d-%m-%Y')})
        #data_bulk.replace(rep)
    else:
        aux=datetime.timedelta(int(j)-1)
        rep.update({j:epoch0+aux})
        #data_bulk.replace({j:epoch0+aux})
        #data_bulk.Date.loc[l_aux]=epoch0+aux
'''

def dater_(j):
    print(j)
    if '/' in j:
        return datetime.datetime.strptime(j,'%d/%m/%Y')
    elif '-' in j:
        return datetime.datetime.strptime(j,'%d-%m-%Y')
    else:
        aux=datetime.timedelta(int(j)-1)
        return epoch0+aux


print('replacing the dates')
data_bulk.Date=data_bulk.Date.apply(dater_)

#data_bulk=data_bulk[data_bulk.Date!='#N/A Value [cd"DtF] for parameter [PERIOD] is not a valid enum']


'''
l_feats=list(set(data_bulk.Feature))

d_feats={}
for j in l_feats:
    aux=data_bulk[data_bulk.Feature==j].pivot_table(index='Date',columns='Ticker',values='Value').fillna(method='ffill')
    d_feats.update({j:aux})
    d_feats[j].fillna(method='ffill',inplace=True)

#d_feats['EPS']=

'''



print('cargango el primer bulk')

data_bulk.to_csv('./data_bulk.txt')


engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                       .format(user="mesa_inversiones",
                               pw="eltata73",
                               host="172.20.15.189",
                               db="unoafp_db"))


print('cargando base')
data_bulk.to_sql('tabla_maestra_javier', con = engine, if_exists = 'replace', index=False)

txt1 = data_bulk.groupby("Ticker")["Feature"].count()
                 

sys.exit('Chupalo')

data_bulk2=pd.read_csv('./your_data_funds.txt')


l_dates2=data_bulk2.Date.unique()

for j in l_dates2:
    print(j)
    l_aux=data_bulk2[data_bulk2.Date==j].index
    if '/' in j:
        data_bulk2.Date.loc[l_aux]=datetime.datetime.strptime(j,'%d/%m/%Y')
    elif '-' in j:
        data_bulk2.Date.loc[l_aux]=datetime.datetime.strptime(j,'%d-%m-%Y')
    else:
        aux=datetime.timedelta(int(j))
        data_bulk2.Date.loc[l_aux]=epoch0+aux

l_feats=list(set(data_bulk2.Feature))

'''
d_feats={}
for j in l_feats:
    aux=data_bulk[data_bulk.Feature==j].pivot_table(index='Date',columns='Ticker',values='Value').fillna(method='ffill')
    d_feats.update({j:aux})
    d_feats[j].fillna(method='ffill',inplace=True)
'''

#d_feats['EPS']=

#sys.exit('chupalo')

data_bulk2.to_csv('./data_bulk_funds.txt')























