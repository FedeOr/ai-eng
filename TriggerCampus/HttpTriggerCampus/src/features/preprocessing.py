import numpy as np
import pandas as pd
import datetime
import os

def dataset_transform(df_log):
    # Valore di TelegramType da mantenere
    ttype = 'write      '
    # Creazione dizionario
    d = {'W':1, '%':2, 'ppm':3, 'C°':4, 'Wh':5, 'bool':6}
    # Rimozione righe con TelegramType = 'response'
    # df_log = df_log.loc[df_log.TelegramType==ttype]
    # Trasformo i dati boolean in modo da poter essere interpretati dal modello
    df_log.loc[df_log.Value=='False',["Value"]] = 0
    df_log.loc[df_log.Value=='True',["Value"]] = 1
    # Creazione della colonna a partire dal mapping del dizionario
    df_log['IdType'] = df_log['ValueType'].map(d)
    # Conversione colonna 'Data'
    df_log['Data'] = pd.to_datetime(df_log['Data'])
    # Conversione colonna 'Value'
    df_log["Value"] = pd.to_numeric(np.char.replace(df_log["Value"].to_numpy().astype(str),',','.'))
    
    return df_log

def preprocessing(df_log):
    
    # Estraggo l'ora dalla data della rilevazione
    df_log['Hour'] = pd.DatetimeIndex(df_log['Data']).hour
    # Rimozione colonne non necessarie per il training del modello
    column_to_drop = ['Data','IndividualAddress','GroupAddress','TelegramType','ValueType','Description']
    df_log.drop(column_to_drop, axis=1, inplace=True)
    # df_log.drop(column_to_drop)
    
    return df_log