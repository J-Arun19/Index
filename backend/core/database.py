import os
import logging
from backend.services.supabase_service import SupabaseService
from backend.services.database_service import DatabaseService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatabaseWrapper")

_use_supabase = False
try:
    SupabaseService.initialize()
    if SupabaseService.verify_connection():
        _use_supabase = True
        DatabaseService.initialize_storage()
        logger.info("Database Wrapper: Supabase backend connected and active.")
    else:
        logger.warning("Database Wrapper: Supabase validation failed. Using SQLite fallback.")
except Exception as e:
    logger.warning(f"Database Wrapper: Supabase failed to initialize ({str(e)}). Using SQLite fallback.")

import sqlite3

def get_db_path():
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(backend_dir, 'users.db')

def get_db():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    # Initialize SQLite for fallback
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            user_name TEXT,
            file_name TEXT NOT NULL,
            media_type TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            report_path TEXT,
            file_path TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS monitoring_sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            user_name TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            status TEXT DEFAULT 'active',
            frames_received INTEGER DEFAULT 0,
            frames_analyzed INTEGER DEFAULT 0,
            frames_skipped INTEGER DEFAULT 0,
            deepfake_detections INTEGER DEFAULT 0,
            average_confidence REAL DEFAULT 0.0,
            average_latency_ms REAL DEFAULT 0.0,
            final_verdict TEXT DEFAULT 'REAL',
            final_risk TEXT DEFAULT 'LOW',
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS monitoring_events (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            trigger_reason TEXT,
            FOREIGN KEY(session_id) REFERENCES monitoring_sessions(id)
        )
    ''')
    conn.commit()
    conn.close()

def create_user(name, email, password_hash):
    if _use_supabase:
        res = DatabaseService.create_user(name, email, password_hash)
        if res is not None:
            return res
        logger.warning("Supabase user creation failed, falling back to SQLite.")

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)',
            (name, email, password_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_email(email):
    if _use_supabase:
        user = DatabaseService.get_user_by_email(email)
        if user is not None:
            return user
        logger.warning("Supabase user email lookup failed, searching SQLite.")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def save_analysis(analysis_id, user_id, user_name, file_name, media_type, prediction, confidence, report_path):
    if _use_supabase:
        try:
            # Upload the media file to Supabase uploads bucket
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            local_path = os.path.join(backend_dir, 'uploads', file_name)
            
            # Save original file inside the uploads bucket with folders structure: images/ or videos/
            folder = "images" if media_type.lower() == "image" else "videos"
            extension = os.path.splitext(file_name)[1]
            destination_path = f"{folder}/{analysis_id}{extension}"
            
            supabase_file_path = None
            if os.path.exists(local_path):
                supabase_file_path = SupabaseService.upload_media(local_path, destination_path)
            
            # Upload the report to reports bucket
            supabase_report_path = None
            if report_path and os.path.exists(report_path):
                report_filename = f"report_{analysis_id}.pdf"
                supabase_report_path = SupabaseService.upload_report(report_path, report_filename)
                
                # Fetch public report URL to save report metadata
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

            # Save the main analysis history record
            res = DatabaseService.save_analysis(
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
            if res:
                return True
        except Exception as e:
            logger.error(f"Supabase save_analysis / storage upload failed: {str(e)}. Falling back to SQLite.")

    # SQLite fallback
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''INSERT INTO analysis_history 
               (id, user_id, user_name, file_name, media_type, prediction, confidence, report_path) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (analysis_id, user_id, user_name, file_name, media_type, prediction, confidence, report_path)
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Error saving analysis: {e}")
        return False
    finally:
        conn.close()

def get_analyses_by_user(user_id):
    if _use_supabase:
        try:
            u_id = int(user_id)
            history = DatabaseService.get_analysis_history(u_id)
            if history:
                return history
        except Exception as e:
            logger.warning(f"Supabase lookup for user {user_id} failed ({str(e)}). Querying SQLite.")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM analysis_history WHERE user_id = ? ORDER BY timestamp DESC',
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def create_monitoring_session(session_id, user_id, user_name):
    if _use_supabase:
        res = DatabaseService.create_monitoring_session(session_id, user_id, user_name)
        if res:
            return True

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''INSERT INTO monitoring_sessions (id, user_id, user_name) VALUES (?, ?, ?)''',
            (session_id, user_id, user_name)
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error creating monitoring session: {e}")
        return False
    finally:
        conn.close()

def update_monitoring_session(session_id, stats):
    if _use_supabase:
        res = DatabaseService.update_monitoring_session(session_id, stats)
        if res:
            return True

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''UPDATE monitoring_sessions SET 
               ended_at = CURRENT_TIMESTAMP,
               status = 'completed',
               frames_received = ?,
               frames_analyzed = ?,
               frames_skipped = ?,
               deepfake_detections = ?,
               average_confidence = ?,
               average_latency_ms = ?,
               final_verdict = ?,
               final_risk = ?
               WHERE id = ?''',
            (
                stats.get('frames_received', 0),
                stats.get('frames_analyzed', 0),
                stats.get('frames_skipped', 0),
                stats.get('deepfake_detections', 0),
                stats.get('average_confidence', 0.0),
                stats.get('average_latency_ms', 0.0),
                stats.get('final_verdict', 'REAL'),
                stats.get('final_risk', 'LOW'),
                session_id
            )
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error updating monitoring session: {e}")
        return False
    finally:
        conn.close()

def log_monitoring_event(session_id, event_id, verdict, confidence, risk_level, latency_ms, trigger_reason):
    if _use_supabase:
        res = DatabaseService.log_monitoring_event(session_id, event_id, verdict, confidence, risk_level, latency_ms, trigger_reason)
        if res:
            return True

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''INSERT INTO monitoring_events (id, session_id, verdict, confidence, risk_level, latency_ms, trigger_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (event_id, session_id, verdict, confidence, risk_level, latency_ms, trigger_reason)
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Error logging monitoring event: {e}")
        return False
    finally:
        conn.close()

def get_user_monitoring_sessions(user_id):
    if _use_supabase:
        try:
            sessions = DatabaseService.get_user_monitoring_sessions(int(user_id))
            if sessions:
                return sessions
        except Exception as e:
            logger.warning(f"Supabase monitoring session lookup failed: {e}")

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM monitoring_sessions WHERE user_id = ? ORDER BY started_at DESC', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

