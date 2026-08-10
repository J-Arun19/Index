import os
import time
from backend.config import Config
from backend.core.context import AnalysisContext
from backend.agents.media_validation_agent import MediaValidationAgent
from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.agents.confidence_evaluation_agent import ConfidenceEvaluationAgent
from backend.agents.fusion_agent import FusionAgent
from backend.agents.explanation_agent import ExplanationAgent
from backend.utils.report_generator import ReportGenerator
from backend.core.database import save_analysis
from backend.utils.logger import setup_logger

logger = setup_logger("Orchestrator")

class Orchestrator:
    def __init__(self):
        self.validation_agent = MediaValidationAgent()
        self.image_agent = ImageAgent()
        self.video_agent = VideoAgent()
        self.confidence_agent = ConfidenceEvaluationAgent()
        self.fusion_agent = FusionAgent()
        self.explanation_agent = ExplanationAgent()

    def locate_file(self, file_id):
        """
        Locates the file in the upload directory that matches the given file_id prefix.
        """
        if not os.path.exists(Config.UPLOAD_FOLDER):
            return None
            
        prefix = f"{file_id}_"
        for filename in os.listdir(Config.UPLOAD_FOLDER):
            if filename.startswith(prefix):
                return os.path.join(Config.UPLOAD_FOLDER, filename)
        return None

    def run_orchestration(self, file_id, media_type, user=None):
        """
        Keeps backward compatibility with previous interface, but routes to the rich context workflow.
        """
        context = AnalysisContext(analysis_id=file_id, user=user, media_type=media_type)
        final_context = self.run_orchestration_context(context)
        return final_context.to_dict()

    def run_orchestration_context(self, context: AnalysisContext) -> AnalysisContext:
        """
        Coordinates the Agentic AI workflow using a shared AnalysisContext.
        """
        # 1. Locate file
        file_path = self.locate_file(context.analysis_id)
        if not file_path:
            raise FileNotFoundError(f"No file found with ID: {context.analysis_id}")
            
        context.uploaded_file = os.path.basename(file_path)

        # Initialize database analysis record (Supabase + SQLite fallback)
        user_id = context.user.get("id") if context.user else None
        user_name = context.user.get("name") if context.user else "Guest"
        try:
            from backend.services.database_service import DatabaseService
            DatabaseService.create_analysis_record(
                analysis_id=context.analysis_id,
                user_id=user_id,
                user_name=user_name,
                file_name=context.uploaded_file,
                media_type=context.media_type
            )
        except Exception as e:
            logger.error(f"Failed to initialize analysis record in DB: {e}")


        # 2. Media Validation Agent
        start = time.time()
        try:
            val_res = self.validation_agent.validate(file_path)
            context.validation_result = val_res
            status = "Success"
            if val_res.get("status") == "invalid":
                status = "Failed"
                context.add_log("Media Validation Agent", status, start, time.time())
                raise ValueError(f"Media validation failed: {', '.join(val_res.get('issues', []))}")
        except Exception as e:
            context.add_log("Media Validation Agent", "Failed", start, time.time())
            raise e
        context.add_log("Media Validation Agent", status, start, time.time())

        # 3. Select specialised analysis agent
        media_type = context.media_type.lower()
        agent_name = "Image Agent" if media_type == "image" else "Video Agent"
        
        start = time.time()
        try:
            if media_type == 'image':
                agent_res = self.image_agent.analyze(file_path)
                context.image_agent_result = agent_res
            elif media_type == 'video':
                agent_res = self.video_agent.analyze(file_path)
                context.video_agent_result = agent_res
            else:
                raise ValueError(f"Unsupported media type: {context.media_type}")
            status = "Success"
        except Exception as e:
            context.add_log(agent_name, "Failed", start, time.time())
            raise e
        context.add_log(agent_name, status, start, time.time())

        # 4. Confidence Evaluation Agent
        start = time.time()
        try:
            conf_res = self.confidence_agent.evaluate(agent_res)
            context.confidence_result = conf_res
            status = "Success"
        except Exception as e:
            context.add_log("Confidence Evaluation Agent", "Failed", start, time.time())
            raise e
        context.add_log("Confidence Evaluation Agent", status, start, time.time())

        # 5. Fusion Agent (Decision Maker)
        start = time.time()
        try:
            fusion_res = self.fusion_agent.fuse(agent_name.replace(" ", ""), agent_res, analysis_id=context.analysis_id)

            context.fusion_result = fusion_res
            status = "Success"
        except Exception as e:
            context.add_log("Fusion Agent", "Failed", start, time.time())
            raise e
        context.add_log("Fusion Agent", status, start, time.time())

        # 6. Explanation Agent
        start = time.time()
        try:
            explanation_res = self.explanation_agent.explain(
                verdict=fusion_res.get("verdict"),
                confidence_level=conf_res.get("level"),
                media_type=media_type,
                features=agent_res.get("features", {})
            )
            context.explanation = explanation_res
            
            # Enrich fusion explanation to include rich details in the UI
            fusion_res["confidence_level"] = conf_res.get("level")
            fusion_res["evidence"] = explanation_res.get("evidence", [])
            fusion_res["recommendation"] = explanation_res.get("recommendation", "N/A")
            status = "Success"
        except Exception as e:
            context.add_log("Explanation Agent", "Failed", start, time.time())
            raise e
        context.add_log("Explanation Agent", status, start, time.time())

        # 7. PDF Report Generator
        start = time.time()
        try:
            report_path = ReportGenerator.generate_report(context.to_dict())
            context.report_path = report_path
            # Build clean report URL for the client
            fusion_res["report_url"] = f"/api/analysis/report/{context.analysis_id}"
            status = "Success"
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            context.add_log("PDF Report Generator", "Failed", start, time.time())
        else:
            context.add_log("PDF Report Generator", status, start, time.time())

        # 8. Save Analysis History
        start = time.time()
        try:
            user_id = context.user.get("id") if context.user else None
            user_name = context.user.get("name") if context.user else "Guest"
            verdict = fusion_res.get("verdict")
            confidence = float(conf_res.get("confidence", 0))
            
            save_analysis(
                analysis_id=context.analysis_id,
                user_id=user_id,
                user_name=user_name,
                file_name=context.uploaded_file,
                media_type=context.media_type,
                prediction=verdict,
                confidence=confidence,
                report_path=context.report_path
            )
            status = "Success"
        except Exception as e:
            logger.error(f"Database save analysis failed: {e}")
            context.add_log("Save Analysis History", "Failed", start, time.time())
        else:
            context.add_log("Save Analysis History", status, start, time.time())

        # Inject logs into fusion results so timeline renders in UI
        fusion_res["timeline"] = context.agent_logs

        return context
