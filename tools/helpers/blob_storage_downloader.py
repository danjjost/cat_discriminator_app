from azure.storage.blob import BlobServiceClient, ContainerClient, BlobProperties
import os

class BlobStorageDownloader:
    def __init__(self, connection_string: str):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    def download_to_directory(self, container_name: str, class_output_dir: str):
        os.makedirs(class_output_dir, exist_ok=True)

        container_client = self.blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs()

        print(f"Downloading blobs from container '{container_name}' into '{class_output_dir}'...")
        
        for blob in blob_list:
            self.__download_blob(blob, class_output_dir, container_client)
            
            
    def __download_blob(self, blob: BlobProperties, class_output_dir: str, container_client: ContainerClient):
        blob_client = container_client.get_blob_client(blob)

        download_file_path = os.path.join(class_output_dir, blob.name)

        with open(download_file_path, "wb") as download_file:
            download_data = blob_client.download_blob()
            download_file.write(download_data.readall())

        print(f"Downloaded {blob.name} to {download_file_path}")
        
    def clear_all_blobs(self, container_name: str):
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_list = container_client.list_blobs()

        print(f"Deleting blobs from container '{container_name}'...")
        
        for blob in blob_list:
            self.__delete_blob(blob, container_client)
            
    def __delete_blob(self, blob: BlobProperties, container_client: ContainerClient):
        blob_client = container_client.get_blob_client(blob)
        blob_client.delete_blob()
        print(f"Deleted {blob.name}")