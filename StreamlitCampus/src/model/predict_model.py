import pandas as pd
import requests
from load_settings import get_setting

def score_model(data: pd.DataFrame):
  
    model_url = get_setting("MODEL_URL")
    headers = {'Authorization': f'Bearer {get_setting("DATABRICKS_TOKEN")}'}
    
    data_json = data.to_dict(orient='split')
    response = requests.request(method='POST', headers=headers, url=model_url, json=data_json)

    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()