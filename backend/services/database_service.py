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

            # Route integer user ID to user_id (FK to public.users), and UUID to owner_id
            db_user_id = None
            db_owner_id = None
            if isinstance(user_id, int) or (isinstance(user_id, str) and user_id.isdigit()):
                db_user_id = int(user_id)
            elif user_id:
                db_owner_id = str(user_id)

            data = {
                "id": analysis_id,
                "user_id": db_user_id,
                "owner_id": db_owner_id,
                "user_name": user_name,
                "file_name": file_name,
                "media_type": media_type,
                "prediction": prediction,
                "confidence": confidence,
                "report_path": report_path
            }

            # Attempt upsert with file_path. Fall back if the column doesn't exist yet or if foreign key is violated.
            try:
                data["file_path"] = file_path
                client.table("analysis_history").upsert(data).execute()
            except Exception as e:
                err_str = str(e)
                if "23503" in err_str or "violates foreign key constraint" in err_str:
                    # Unsynced profile in public.users: preserve analysis data by nullifying foreign keys
                    data["user_id"] = None
                    data["owner_id"] = None
                data.pop("file_path", None)
                try:
                    client.table("analysis_history").upsert(data).execute()
                except Exception as e2:
                    if "23503" in str(e2) or "violates foreign key constraint" in str(e2):
                        data["user_id"] = None
                        data["owner_id"] = None
                        client.table("analysis_history").upsert(data).execute()
                    else:
                        raise e2

            logger.info(f"Saved completed analysis {analysis_id} to Supabase database.")
            return True
        except Exception as e:
            logger.error(f"Failed to save analysis record to Supabase: {str(e)}")
            return False

    @classmethod
    def get_analysis_history(cls, user_id: Any) -> List[Dict[str, Any]]:
        """
        Retrieves the analysis runs history for a specific user, ordered chronologically.
        Supports integer user_id or string UUID owner_id.
        """
        try:
            client = SupabaseService.get_client()
            if isinstance(user_id, int) or (isinstance(user_id, str) and user_id.isdigit()):
                res = client.table("analysis_history").select("*").eq("user_id", int(user_id)).order("timestamp", desc=True).execute()
            else:
                res = client.table("analysis_history").select("*").eq("owner_id", str(user_id)).order("timestamp", desc=True).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"Failed to fetch analysis history for user {user_id}: {str(e)}")
            return []

    @classmethod
    def get_analysis_by_id(cls, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single analysis run record by its ID.
        """
        try:
            client = SupabaseService.get_client()
            res = client.table("analysis_history").select("*").eq("id", analysis_id).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to fetch analysis {analysis_id}: {str(e)}")
            return None

    @classmethod
    def get_agent_logs(cls, analysis_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves agent execution trace logs for a specific analysis run.
        """
        try:
            client = SupabaseService.get_client()
            res = client.table("agent_logs").select("*").eq("analysis_id", analysis_id).order("created_at", desc=False).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"Failed to fetch agent logs for analysis {analysis_id}: {str(e)}")
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

    @classmethod
    def create_monitoring_session(cls, session_id: str, user_id: Optional[int], user_name: str) -> bool:
        """
        Creates a new monitoring session in Supabase PostgreSQL.
        """
        try:
            client = SupabaseService.get_client()
            data = {
                "id": session_id,
                "user_id": user_id,
                "user_name": user_name,
                "status": "active",
                "frames_received": 0,
                "frames_analyzed": 0,
                "frames_skipped": 0,
                "deepfake_detections": 0,
                "average_confidence": 0.0,
                "average_latency_ms": 0.0,
                "final_verdict": "REAL",
                "final_risk": "LOW"
            }
            client.table("monitoring_sessions").insert(data).execute()
            logger.info(f"Created monitoring session in Supabase: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create monitoring session in Supabase: {str(e)}")
            return False

    @classmethod
    def update_monitoring_session(cls, session_id: str, stats: Dict[str, Any]) -> bool:
        """
        Updates an existing monitoring session with final aggregated statistics.
        """
        try:
            client = SupabaseService.get_client()
            stats["status"] = "completed"
            client.table("monitoring_sessions").update(stats).eq("id", session_id).execute()
            logger.info(f"Updated monitoring session in Supabase: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update monitoring session in Supabase: {str(e)}")
            return False

    @classmethod
    def log_monitoring_event(cls, session_id: str, event_id: str, verdict: str, confidence: float, risk_level: str, latency_ms: float, trigger_reason: str) -> bool:
        """
        Logs a meaningful suspicious/deepfake detection event for a monitoring session.
        """
        try:
            client = SupabaseService.get_client()
            client.table("monitoring_events").insert({
                "id": event_id,
                "session_id": session_id,
                "verdict": verdict,
                "confidence": confidence,
                "risk_level": risk_level,
                "latency_ms": latency_ms,
                "trigger_reason": trigger_reason
            }).execute()
            logger.info(f"Logged monitoring event '{event_id}' for session '{session_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to log monitoring event in Supabase: {str(e)}")
            return False

    @classmethod
    def get_user_monitoring_sessions(cls, user_id: int) -> List[Dict[str, Any]]:
        """
        Fetches all monitoring sessions for a specific user.
        """
        try:
            client = SupabaseService.get_client()
            res = client.table("monitoring_sessions").select("*").eq("user_id", user_id).order("started_at", desc=True).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"Failed to fetch monitoring sessions for user {user_id}: {str(e)}")
            return []

