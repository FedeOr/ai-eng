import os
from azure.storage.blob import BlobServiceClient, BlobClient

def save_json(json_data: str, filename: str):
    conn_string = os.environ.get("AZURE_STORAGE_ACCOUNT")
    container = os.environ.get("BLOB_CONTAINER_NAME")
    
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    blob_client = blob_service_client.get_blob_client(container=container, blob=filename)
    blob_client.upload_blob(json_data)