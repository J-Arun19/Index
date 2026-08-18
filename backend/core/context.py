import time

class AnalysisContext:
    def __init__(self, analysis_id: str, user: dict = None, uploaded_file: str = None, media_type: str = None):
        self.analysis_id = analysis_id
        self.user = user  # E.g. {"id": 1, "name": "Admin", "email": "admin@example.com"} or None
        self.uploaded_file = uploaded_file
        self.media_type = media_type  # "image" or "video"
        
        self.validation_result = None
        self.image_agent_result = None
        self.video_agent_result = None
        self.metadata_agent_result = None
        self.artifact_agent_result = None
        self.confidence_result = None
        self.fusion_result = None
        self.explanation = None
        self.report_path = None
        self.decision_history = []
        self.active_agents = []
        self.failures = []
        self.timestamp = time.time()
        self.agent_logs = []  # List of logs for each agent execution

    def add_log(self, agent_name: str, status: str, start_time: float, end_time: float):
        duration = end_time - start_time
        rounded_duration = round(duration, 4)
        self.agent_logs.append({
            "agent": agent_name,
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "duration": rounded_duration
        })
        try:
            # Dynamically insert log into Supabase database layer
            from backend.services.database_service import DatabaseService
            DatabaseService.save_agent_log(
                analysis_id=self.analysis_id,
                agent_name=agent_name,
                status=status,
                duration=rounded_duration
            )
        except Exception:
            pass


    def to_dict(self) -> dict:
        return {
            "analysis_id": self.analysis_id,
            "user": self.user,
            "uploaded_file": self.uploaded_file,
            "media_type": self.media_type,
            "validation_result": self.validation_result,
            "image_agent_result": self.image_agent_result,
            "video_agent_result": self.video_agent_result,
            "metadata_agent_result": self.metadata_agent_result,
            "artifact_agent_result": self.artifact_agent_result,
            "confidence_result": self.confidence_result,
            "fusion_result": self.fusion_result,
            "explanation": self.explanation,
            "report_path": self.report_path,
            "decision_history": self.decision_history,
            "active_agents": self.active_agents,
            "failures": self.failures,
            "timestamp": self.timestamp,
            "agent_logs": self.agent_logs
        }
