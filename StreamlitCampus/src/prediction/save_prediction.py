from azure.storage.blob import BlobServiceClient, BlobClient

def save_json(json_data: str, filename: str):
    conn_string ="DefaultEndpointsProtocol=https;AccountName=storageaccountaieng92cc;AccountKey=yRee7zuNsdVCv2AFf/MhrGd8eGfOPncQvDfmXYN3F9/wQ9QCj+9RE0k1r+kXtehudbWDNgZ+3cQqGKFVivgWKg==;EndpointSuffix=core.windows.net"#"AZURE_STORAGE_ACCOUNT"
    container = "team2"#("BLOB_CONTAINER_NAME"
    
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    blob_client = blob_service_client.get_blob_client(container=container, blob=filename)
    blob_client.upload_blob(json_data)