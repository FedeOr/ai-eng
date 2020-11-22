import logging

import azure.functions as func
import pandas as pd
from io import StringIO

def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")
    
    text = myblob.read().decode('utf-8')
    df = pd.read_json(StringIO(text), orient='split')
    
    logging.info(f"Shape:{df.shape}")
