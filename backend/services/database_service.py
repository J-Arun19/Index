import os
import logging
from typing import Dict, Any, List, Optional
from backend.services.supabase_service import SupabaseService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatabaseService")

class DatabaseService:
    """
    Service layer providing CRUD operations and file storage access for Supabase integration.
    """

    @classmethod
    def initialize_storage(cls) -> None:
        """
        Ensures the 'uploads' and 'reports' storage buckets exist and are public.
        """
        try:
            client = SupabaseService.get_client()
            for bucket in ["uploads", "reports"]:
                try:
                    client.storage.create_bucket(bucket, options={"public": True})
                    logger.info(f"Configured Supabase storage bucket: '{bucket}'")
                except Exception as e:
                    # Supabase will return error if already exists, which is safe to ignore
                    logger.info(f"Bucket '{bucket}' already exists or creation skipped: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to initialize storage buckets: {str(e)}")

    @classmethod
    def create_user(cls, name: str, email: str, password_hash: str) -> Optional[int]:
        """
        Creates a new user profile record in PostgreSQL.
        Returns the new user's ID if successful, otherwise None.
        """
        try:
            client = SupabaseService.get_client()
            res = client.table("users").insert({
                "name": name,
                "email": email,
                "password_hash": password_hash
            }).execute()
            
            if res.data and len(res.data) > 0:
                user_id = res.data[0]["id"]
                logger.info(f"Created user in PostgreSQL: {email} (ID: {user_id})")
                return user_id
            return None
        except Exception as e:
            logger.error(f"Failed to create user in Supabase PostgreSQL: {str(e)}")
            return None

    @classmethod
    def get_user_by_email(cls, email: str) -> Optional[Dict[str, Any]]:
        """
        Fetches a user profile from PostgreSQL by email address.
        """
        try:
            client = SupabaseService.get_client()
            res = client.table("users").select("*").eq("email", email).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to fetch user by email: {str(e)}")
            return None

    @classmethod
    def get_user(cls, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetches a user profile from PostgreSQL by ID.
        """
        try:
            client = SupabaseService.get_client()
            res = client.table("users").select("*").eq("id", user_id).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to fetch user by ID: {str(e)}")
            return None

    @classmethod
    def create_analysis_record(
        cls,
        analysis_id: str,
        user_id: Optional[int],
        user_name: str,
        file_name: str,
        media_type: str
    ) -> bool:
        """
        Initializes an analysis record in the PostgreSQL database at the start of orchestration.
        """
        try:
            client = SupabaseService.get_client()
            folder = "images" if media_type.lower() == "image" else "videos"
            extension = os.path.splitext(file_name)[1]
            bucket = os.getenv("SUPABASE_UPLOADS_BUCKET", "uploads")
            file_path = f"{bucket}/{folder}/{analysis_id}{extension}"

            data = {
                "id": analysis_id,
                "user_id": user_id,
                "user_name": user_name,
                "file_name": file_name,
                "media_type": media_type,
                "prediction": "PROCESSING",
                "confidence": 0.0
            }
            
            # Attempt insert with file_path. Fall back if the column doesn't exist yet.
            try:
                data["file_path"] = file_path
                client.table("analysis_history").insert(data).execute()
            except Exception as e:
                logger.warning(f"Failed to insert with file_path: {str(e)}. Retrying without file_path...")
                data.pop("file_path", None)
                client.table("analysis_history").insert(data).execute()

            logger.info(f"Created analysis history record: {analysis_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize analysis record in Supabase: {str(e)}")
            return False

    @classmethod
    def update_fusion_prediction(cls, analysis_id: str, prediction: str, confidence: float) -> bool:
        """
        Updates the final prediction and confidence evaluation results of an analysis.
        """
        try:
            client = SupabaseService.get_client()
            client.table("analysis_history").update({
                "prediction": prediction,
                "confidence": confidence
            }).eq("id", analysis_id).execute()
            logger.info(f"Updated prediction results for analysis '{analysis_id}': {prediction} ({confidence}%)")
            return True
        except Exception as e:
            logger.error(f"Failed to update fusion prediction in Supabase: {str(e)}")
            return False

    @classmethod
    def save_analysis(
        cls,
        analysis_id: str,
        user_id: Optional[int],
        user_name: str,
        file_name: str,
        media_type: str,
        prediction: str,
        confidence: float,
        report_path: Optional[str],
        file_path: Optional[str] = None
    ) -> bool:
        """
        Saves or updates a complete analysis run record.
        """
        try:
            client = SupabaseService.get_client()
            if not file_path:
                folder = "images" if media_type.lower() == "image" else "videos"
                extension = os.path.splitext(file_name)[1]
                bucket = os.getenv("SUPABASE_UPLOADS_BUCKET", "uploads")
                file_path = f"{bucket}/{folder}/{analysis_id}{extension}"

            data = {
                "id": analysis_id,
                "user_id": user_id,
                "user_name": user_name,
                "file_name": file_name,
                "media_type": media_type,
                "prediction": prediction,
                "confidence": confidence,
                "report_path": report_path
            }

            # Attempt upsert with file_path. Fall back if the column doesn't exist yet.
            try:
                data["file_path"] = file_path
                client.table("analysis_history").upsert(data).execute()
            except Exception as e:
                logger.warning(f"Failed to upsert with file_path: {str(e)}. Retrying without file_path...")
                data.pop("file_path", None)
                client.table("analysis_history").upsert(data).execute()

            logger.info(f"Saved completed analysis {analysis_id} to Supabase database.")
            return True
        except Exception as e:
            logger.error(f"Failed to save analysis record to Supabase: {str(e)}")
            return False

    @classmethod
    def get_analysis_history(cls, user_id: int) -> List[Dict[str, Any]]:
        """
        Retrieves the analysis runs history for a specific user, ordered chronologically.
        """
        try:
            client = SupabaseService.get_client()
            res = client.table("analysis_history").select("*").eq("user_id", user_id).order("timestamp", desc=True).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"Failed to fetch analysis history for user {user_id}: {str(e)}")
            return []

    @classmethod
    def save_agent_log(
        cls,
        analysis_id: str,
        agent_name: str,
        status: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Saves a trace telemetry log for an agent execution stage.
        """
        try:
            client = SupabaseService.get_client()
            client.table("agent_logs").insert({
                "analysis_id": analysis_id,
                "agent_name": agent_name,
                "status": status,
                "duration": duration,
                "metadata": metadata or {}
            }).execute()
            logger.info(f"Recorded agent telemetry: '{agent_name}' ({duration}s) on analysis '{analysis_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to save agent trace log to Supabase: {str(e)}")
            return False

    @classmethod
    def save_report_metadata(
        cls,
        analysis_id: str,
        report_url: str,
        file_size: Optional[int] = None,
        report_path: Optional[str] = None
    ) -> bool:
        """
        Saves report publication metadata in the reports table.
        """
        try:
            client = SupabaseService.get_client()
            data = {
                "analysis_id": analysis_id,
                "report_url": report_url,
                "file_size": file_size
            }
            if report_path:
                data["report_path"] = report_path

            # Attempt upsert with report_path. Fall back if the column doesn't exist yet.
            try:
                client.table("reports").upsert(data).execute()
            except Exception as e:
                if "report_path" in data:
                    logger.warning(f"Failed to upsert with report_path: {str(e)}. Retrying without report_path...")
                    data.pop("report_path", None)
                    client.table("reports").upsert(data).execute()
                else:
                    raise e

            logger.info(f"Recorded report metadata for analysis '{analysis_id}': {report_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to record report metadata to Supabase: {str(e)}")
            return False
