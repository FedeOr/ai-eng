from azure.storage.blob import BlobServiceClient, BlobClient
from load_settings import get_setting

def save_json(json_data: str, filename: str):
    conn_string = get_setting("AZURE_STORAGE_ACCOUNT")
    container = get_setting("BLOB_CONTAINER_NAME")
    
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    blob_client = blob_service_client.get_blob_client(container=container, blob=filename)
    blob_client.upload_blob(json_data)