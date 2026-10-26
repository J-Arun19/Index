import os
import logging
from typing import Dict, Any, List, Optional
from backend.services.supabase_service import SupabaseService
from backend.services.database_service import DatabaseService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatabaseWrapper")

_use_supabase = False
_db_initialized = False

def init_db():
    """
    Initializes the Supabase database connection and storage buckets.
    Supabase is the sole persistent database. SQLite is not used.
    """
    global _use_supabase, _db_initialized
    if _db_initialized:
        return _use_supabase
    try:
        SupabaseService.initialize()
        if SupabaseService.verify_connection():
            _use_supabase = True
            DatabaseService.initialize_storage()
            logger.info("Database Wrapper: Supabase backend connected and active.")
        else:
            _use_supabase = False
            logger.error("Database Wrapper: Supabase connection failed. Please check credentials.")
    except Exception as e:
        _use_supabase = False
        logger.error(f"Database Wrapper: Supabase initialization error: {str(e)}")
    finally:
        _db_initialized = True
    return _use_supabase

def create_user(name: str, email: str, password_hash: str) -> Optional[int]:
    """
    Creates a new user in Supabase PostgreSQL.
    """
    if not _use_supabase:
        logger.error("Supabase unavailable. User creation aborted.")
        return None
    return DatabaseService.create_user(name, email, password_hash)

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user profile from Supabase PostgreSQL by email.
    """
    if not _use_supabase:
        logger.error("Supabase unavailable. User lookup aborted.")
        return None
    return DatabaseService.get_user_by_email(email)

def save_analysis(analysis_id, user_id, user_name, file_name, media_type, prediction, confidence, report_path):
    """
    Saves analysis results and associated media/report metadata in Supabase PostgreSQL & Storage.
    """
    if not _db_initialized:
        init_db()
    if not _use_supabase:
        logger.error("Supabase unavailable. Analysis save aborted.")
        return False

    try:
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path = os.path.join(backend_dir, 'uploads', file_name)
        
        folder = "images" if media_type.lower() == "image" else "videos"
        extension = os.path.splitext(file_name)[1]
        destination_path = f"{folder}/{analysis_id}{extension}"
        
        supabase_file_path = None
        if os.path.exists(local_path):
            supabase_file_path = SupabaseService.upload_media(local_path, destination_path)
        
        supabase_report_path = None
        if report_path and os.path.exists(report_path):
            report_filename = f"report_{analysis_id}.pdf"
            supabase_report_path = SupabaseService.upload_report(report_path, report_filename)
            
            if supabase_report_path:
                bucket = os.getenv("SUPABASE_REPORTS_BUCKET", "reports")
                public_report_url = SupabaseService.get_client().storage.from_(bucket).get_public_url(report_filename)
                file_size = os.path.getsize(report_path)
                DatabaseService.save_report_metadata(
                    analysis_id=analysis_id,
                    report_url=public_report_url,
                    file_size=file_size,
                    report_path=supabase_report_path
                )

        return DatabaseService.save_analysis(
            analysis_id=analysis_id,
            user_id=user_id,
            user_name=user_name,
            file_name=file_name,
            media_type=media_type,
            prediction=prediction,
            confidence=confidence,
            report_path=report_path,
            file_path=supabase_file_path
        )
    except Exception as e:
        logger.error(f"Supabase save_analysis failed: {str(e)}")
        return False

def get_analyses_by_user(user_id):
    """
    Fetches analysis history for a user from Supabase PostgreSQL.
    Supports both legacy integer user IDs and Supabase Auth string UUIDs.
    """
    if not _db_initialized:
        init_db()
    if not _use_supabase:
        logger.error("Supabase unavailable. History lookup aborted.")
        return []
    try:
        return DatabaseService.get_analysis_history(user_id)
    except Exception as e:
        logger.error(f"Supabase analysis lookup failed for user {user_id}: {str(e)}")
        return []


def get_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single analysis run record by its ID.
    """
    if not _db_initialized:
        init_db()
    if not _use_supabase:
        logger.error("Supabase unavailable. Analysis lookup aborted.")
        return None
    try:
        return DatabaseService.get_analysis_by_id(analysis_id)
    except Exception as e:
        logger.error(f"Supabase analysis lookup failed for {analysis_id}: {str(e)}")
        return None

def get_agent_logs(analysis_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves agent logs for an analysis ID.
    """
    if not _db_initialized:
        init_db()
    if not _use_supabase:
        return []
    try:
        return DatabaseService.get_agent_logs(analysis_id)
    except Exception as e:
        logger.error(f"Supabase agent logs lookup failed for {analysis_id}: {str(e)}")
        return []

def create_monitoring_session(session_id: str, user_id: Optional[int], user_name: str) -> bool:
    """
    Creates a monitoring session record in Supabase PostgreSQL.
    """
    if not _use_supabase:
        logger.error("Supabase unavailable. Monitoring session creation aborted.")
        return False
    return DatabaseService.create_monitoring_session(session_id, user_id, user_name)

def update_monitoring_session(session_id: str, stats: Dict[str, Any]) -> bool:
    """
    Updates a monitoring session record in Supabase PostgreSQL.
    """
    if not _use_supabase:
        logger.error("Supabase unavailable. Monitoring session update aborted.")
        return False
    return DatabaseService.update_monitoring_session(session_id, stats)

def log_monitoring_event(session_id: str, event_id: str, verdict: str, confidence: float, risk_level: str, latency_ms: float, trigger_reason: str) -> bool:
    """
    Logs a monitoring detection event in Supabase PostgreSQL.
    """
    if not _use_supabase:
        logger.error("Supabase unavailable. Monitoring event logging aborted.")
        return False
    return DatabaseService.log_monitoring_event(session_id, event_id, verdict, confidence, risk_level, latency_ms, trigger_reason)

def get_user_monitoring_sessions(user_id):
    """
    Fetches user monitoring sessions from Supabase PostgreSQL.
    """
    if not _use_supabase:
        logger.error("Supabase unavailable. Monitoring session lookup aborted.")
        return []
    try:
        return DatabaseService.get_user_monitoring_sessions(int(user_id))
    except Exception as e:
        logger.error(f"Supabase monitoring lookup failed for user {user_id}: {str(e)}")
        return []
