import pandas as pd
import requests

def score_model(data: pd.DataFrame):
    
    model_url = "https://adb-4483624067336826.6.azuredatabricks.net/model/Team2-IsoForest/Production/invocations"#os.environ.get("MODEL_URL")
    headers = {'Authorization': f'Bearer {"dapi183eecc004e8a34a083cc8389d6836b4"}'} #{'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
    data_json = data.to_dict(orient='split')
    response = requests.request(method='POST', headers=headers, url=model_url, json=data_json)

    if response.status_code != 200:
      raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()