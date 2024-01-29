# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:50:57 2019

@author: VictorBorckChirighin
"""
import pandas as pd
import mysql.connector as ms

def descarga_tabla(tabla):
    mydb = ms.connect(host="172.20.15.189", user="mesa_inversiones",
                      passwd="eltata73", database="unoafp_db",
                      use_pure=True)
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM " + tabla)
    df = pd.DataFrame(mycursor.fetchall())
    df.columns = mycursor.column_names
    mycursor.close()
    mydb.close()
    return df

def descarga_query(qry):
    mydb = ms.connect(host="172.20.15.189", user="mesa_inversiones",
                      passwd="eltata73", database="unoafp_db",
                      use_pure=True)
    mycursor = mydb.cursor()
    mycursor.execute(qry)
    df = pd.DataFrame(mycursor.fetchall())
    df.columns = mycursor.column_names
    mycursor.close()
    mydb.close()
    return df

def from_javier_master(l,fill):
    sql='SELECT * FROM tabla_maestra_javier WHERE Ticker in '+str(l)
    df=pd.pivot_table(descarga_query(sql),index='Date',columns='Ticker',values='Value')
    if fill:
        df.fillna(method='ffill',inplace=True)
    return df

def descarga_tabla2(tabla):
    mydb = ms.connect(host="172.20.15.189", user="mesa_inversiones",
                      passwd="eltata73", database="Pruebas",
                      use_pure=True)
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM " + tabla)
    df = pd.DataFrame(mycursor.fetchall())
    df.columns = mycursor.column_names
    mycursor.close()
    mydb.close()
    return df

def descarga_query2(qry):
    mydb = ms.connect(host="172.20.15.189", user="mesa_inversiones",
                      passwd="eltata73", database="Pruebas",
                      use_pure=True)
    mycursor = mydb.cursor()
    mycursor.execute(qry)
    df = pd.DataFrame(mycursor.fetchall())
    df.columns = mycursor.column_names
    mycursor.close()
    mydb.close()
    return df



if __name__ == "__main__":
    tabla = 'clas_fondos'
    df = descarga_tabla(tabla)  # !!!
    
    last = df[df['fecha']==df['fecha'].max()]
    last.to_clipboard(decimal=',')

    nemos = ['US4642868065', 'DE0005933931', 'LU0274211480', 'DE000DWS2D90']
    df['precio_dia'] = df['precio_dia'].astype(float)
    test = df[df['nemotecnico'].isin(nemos)]
    test.to_clipboard(decimal=',')
    df['fecha'].unique()

