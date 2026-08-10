import os
import logging
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SupabaseService")

class SupabaseService:
    _client: Optional[Client] = None

    @classmethod
    def get_client(cls) -> Client:
        """
        Retrieves the initialized Supabase client singleton instance.
        """
        if cls._client is None:
            cls.initialize()
        return cls._client

    @classmethod
    def initialize(cls) -> None:
        """
        Loads the environment and initializes the Supabase client using service role key.
        """
        # Resolve the absolute path of the backend/.env file
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(backend_dir, ".env")
        
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from: {env_path}")
        else:
            load_dotenv()
            logger.warning(f"Dotenv file not found at {env_path}. Using environment-level configuration.")

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        # Validate existence of configuration variables
        if not supabase_url:
            raise ValueError("Configuration Error: SUPABASE_URL is missing.")
        if not supabase_key:
            raise ValueError("Configuration Error: SUPABASE_SERVICE_ROLE_KEY is missing.")

        try:
            cls._client = create_client(supabase_url, supabase_key)
            logger.info("Supabase client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise e

    @classmethod
    def verify_connection(cls) -> bool:
        """
        Checks connectivity by performing a lightweight query (listing storage buckets).
        Returns True if successful, False otherwise.
        """
        try:
            client = cls.get_client()
            # Requesting storage buckets list validates both URL structure and Service Role permissions
            client.storage.list_buckets()
            logger.info("Supabase connectivity verified successfully.")
            return True
        except Exception as e:
            logger.error(f"Supabase connection validation failed: {str(e)}")
            return False

    @classmethod
    def upload_media(cls, local_file_path: str, destination_path: str) -> Optional[str]:
        """
        Uploads local media file to the uploads bucket.
        Returns the storage path (e.g. 'uploads/images/filename.jpg') if successful.
        """
        bucket = os.getenv("SUPABASE_UPLOADS_BUCKET", "uploads")
        try:
            client = cls.get_client()
            with open(local_file_path, "rb") as f:
                file_content = f.read()
            
            client.storage.from_(bucket).upload(
                path=destination_path,
                file=file_content,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
            full_storage_path = f"{bucket}/{destination_path}"
            logger.info(f"Uploaded media to Supabase Storage: {full_storage_path}")
            return full_storage_path
        except Exception as e:
            logger.error(f"Failed to upload media to storage bucket '{bucket}': {str(e)}")
            return None

    @classmethod
    def download_media(cls, storage_path: str, local_destination_path: str) -> bool:
        """
        Downloads a media file from Supabase Storage.
        """
        bucket = os.getenv("SUPABASE_UPLOADS_BUCKET", "uploads")
        try:
            client = cls.get_client()
            path_in_bucket = storage_path
            if path_in_bucket.startswith(f"{bucket}/"):
                path_in_bucket = path_in_bucket[len(bucket) + 1:]
                
            res = client.storage.from_(bucket).download(path_in_bucket)
            os.makedirs(os.path.dirname(os.path.abspath(local_destination_path)), exist_ok=True)
            with open(local_destination_path, "wb") as f:
                f.write(res)
            logger.info(f"Downloaded media {storage_path} to {local_destination_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download media '{storage_path}': {str(e)}")
            return False

    @classmethod
    def delete_media(cls, storage_path: str) -> bool:
        """
        Deletes a media file from Supabase Storage.
        """
        bucket = os.getenv("SUPABASE_UPLOADS_BUCKET", "uploads")
        try:
            client = cls.get_client()
            path_in_bucket = storage_path
            if path_in_bucket.startswith(f"{bucket}/"):
                path_in_bucket = path_in_bucket[len(bucket) + 1:]
                
            client.storage.from_(bucket).remove([path_in_bucket])
            logger.info(f"Deleted media from storage: {storage_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete media '{storage_path}': {str(e)}")
            return False

    @classmethod
    def upload_report(cls, local_file_path: str, destination_path: str) -> Optional[str]:
        """
        Uploads local PDF report to the reports bucket.
        Returns the storage path (e.g. 'reports/report.pdf') if successful.
        """
        bucket = os.getenv("SUPABASE_REPORTS_BUCKET", "reports")
        try:
            client = cls.get_client()
            with open(local_file_path, "rb") as f:
                file_content = f.read()
            
            client.storage.from_(bucket).upload(
                path=destination_path,
                file=file_content,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
            full_storage_path = f"{bucket}/{destination_path}"
            logger.info(f"Uploaded report to Supabase Storage: {full_storage_path}")
            return full_storage_path
        except Exception as e:
            logger.error(f"Failed to upload report to storage bucket '{bucket}': {str(e)}")
            return None

    @classmethod
    def download_report(cls, storage_path: str, local_destination_path: str) -> bool:
        """
        Downloads a report file from Supabase Storage.
        """
        bucket = os.getenv("SUPABASE_REPORTS_BUCKET", "reports")
        try:
            client = cls.get_client()
            path_in_bucket = storage_path
            if path_in_bucket.startswith(f"{bucket}/"):
                path_in_bucket = path_in_bucket[len(bucket) + 1:]
                
            res = client.storage.from_(bucket).download(path_in_bucket)
            os.makedirs(os.path.dirname(os.path.abspath(local_destination_path)), exist_ok=True)
            with open(local_destination_path, "wb") as f:
                f.write(res)
            logger.info(f"Downloaded report {storage_path} to {local_destination_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download report '{storage_path}': {str(e)}")
            return False

    @classmethod
    def delete_report(cls, storage_path: str) -> bool:
        """
        Deletes a report file from Supabase Storage.
        """
        bucket = os.getenv("SUPABASE_REPORTS_BUCKET", "reports")
        try:
            client = cls.get_client()
            path_in_bucket = storage_path
            if path_in_bucket.startswith(f"{bucket}/"):
                path_in_bucket = path_in_bucket[len(bucket) + 1:]
                
            client.storage.from_(bucket).remove([path_in_bucket])
            logger.info(f"Deleted report from storage: {storage_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete report '{storage_path}': {str(e)}")
            return False

