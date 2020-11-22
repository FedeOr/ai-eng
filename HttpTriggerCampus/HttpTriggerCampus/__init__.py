import logging

import azure.functions as func

import pandas as pd
import datetime
import os
#import azure.storage.blob as asb
from azure.storage.blob import BlobServiceClient, BlobClient

def score_model(data: pd.DataFrame):
    
    model_url = os.environ.get("MODEL_URL")
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'} #Aggiungere nelle local.settings
    data_json = data.to_dict(orient='split')
    response = requests.request(method='POST', headers=headers, url=url, json=data_json)

    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

def save_json(json_data: str):
    conn_string = os.environ.get("AZURE_STORAGE_ACCOUNT")
    container = os.environ.get("BLOB_CONTAINER_NAME")
    
    utc_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    json_filename = f"prediction_{utc_timestamp}.json"
    
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    blob_client = blob_service_client.get_blob_client(container=container, blob=json_filename)
    blob_client.upload_blob(json_data)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    status_code = 200

    try:
        df = pd.read_csv("dataset\campus_log.csv", sep=',')

        logging.debug(df.head())
        
        #prediction = score_model(df)
        #df['prediction'] = prediction
        
        json_result = df.to_json(orient='split')
        
        save_json(json_result)
        
        return func.HttpResponse('Prediction file created', status_code = status_code)
    
    except Exception as e:
        status_code = 400
        return func.HttpResponse(f'Error: {e}', status_code = status_code)