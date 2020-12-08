import logging
import azure.functions as func
import pandas as pd
import numpy as np
import datetime
import os
import pyodbc
import requests

from azure.storage.blob import BlobServiceClient, BlobClient

def save_json(json_data: str, filename: str):
    conn_string = os.environ.get("AZURE_STORAGE_ACCOUNT")
    container = os.environ.get("BLOB_CONTAINER_NAME")
    
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    blob_client = blob_service_client.get_blob_client(container=container, blob=filename)
    blob_client.upload_blob(json_data)

def score_model(data: pd.DataFrame):
  
    model_url = os.environ.get("MODEL_URL")
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
    
    data_json = data.to_dict(orient='split')
    response = requests.request(method='POST', headers=headers, url=model_url, json=data_json)

    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

def read_DB_table(tableName):
    
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                      'Server='+os.environ.get("SERVER")+';'
                      'Database='+os.environ.get("DATABASE")+';'
                      'Uid='+os.environ.get("DB_USER")+';'
                      'Pwd='+os.environ.get("DB_PWD")+';'
                      'Connection Timeout=30')

    cursor = conn.cursor()

    df_database = pd.read_sql_query('SELECT * FROM [dbo].['+ tableName+']', conn)
    
    return df_database

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

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    status_code = 200
    table_name = req.params.get('table_name')
    json_filename = req.params.get('json_filename')

    try:
        df = read_DB_table(table_name)
                    
        df_transform = dataset_transform(df)
        df_preprocess = preprocessing(df_transform)
        prediction = score_model(df_preprocess)
        df['Prediction'] = prediction
        
        json_result = df.to_json(orient='split')
        save_json(json_result, json_filename)
        
        return func.HttpResponse(json_result, status_code = status_code)
    
    except Exception as e:
        status_code = 400
        return func.HttpResponse(f'Error: {e}', status_code = status_code)