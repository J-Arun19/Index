import os
import time
from backend.config import Config
from backend.core.context import AnalysisContext
from backend.agents.media_validation_agent import MediaValidationAgent
from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.agents.metadata_agent import MetadataAgent
from backend.agents.artifact_agent import ArtifactConsistencyAgent
from backend.agents.confidence_evaluation_agent import ConfidenceEvaluationAgent
from backend.agents.fusion_agent import FusionAgent
from backend.agents.explanation_agent import ExplanationAgent
from backend.utils.report_generator import ReportGenerator
from backend.core.database import save_analysis
from backend.utils.logger import setup_logger
from backend.utils.config import PipelineConfig

logger = setup_logger("Orchestrator")

class Orchestrator:
    def __init__(self):
        self.validation_agent = MediaValidationAgent()
        self.image_agent = ImageAgent()
        self.video_agent = VideoAgent()
        self.metadata_agent = MetadataAgent()
        self.artifact_agent = ArtifactConsistencyAgent()
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

    def run_orchestration(self, file_id, media_type=None, user=None, selected_providers=None):
        """
        Keeps backward compatibility with previous interface, but routes to the rich context workflow.
        """
        context = AnalysisContext(analysis_id=file_id, user=user, media_type=media_type, selected_providers=selected_providers)
        final_context = self.run_orchestration_context(context)
        return final_context.to_dict()

    def run_orchestration_context(self, context: AnalysisContext) -> AnalysisContext:
        """
        Coordinates the Agentic AI workflow using a shared AnalysisContext.
        Autonomously detects media type and routes to Image Agent or Video Agent.
        """
        # 1. Locate file
        file_path = self.locate_file(context.analysis_id)
        if not file_path:
            raise FileNotFoundError(f"No file found with ID: {context.analysis_id}")
            
        context.uploaded_file = os.path.basename(file_path)

        # 2. Media Validation Agent & Media Type Detection
        start = time.time()
        try:
            # Autonomous validation: detect media type directly from content signature
            val_res = self.validation_agent.validate(file_path, expected_media_type=None)
            context.validation_result = val_res
            
            if not val_res.get("valid", False):
                status = "Failed"
                context.add_log("Media Validation Agent", status, start, time.time())
                
                # Save the failure to analysis history database
                try:
                    user_id = context.user.get("id") if context.user else None
                    user_name = context.user.get("name") if context.user else "Guest"
                    save_analysis(
                        analysis_id=context.analysis_id,
                        user_id=user_id,
                        user_name=user_name,
                        file_name=context.uploaded_file,
                        media_type=context.media_type or "unknown",
                        prediction="FAILED_VALIDATION",
                        confidence=0.0,
                        report_path=None
                    )
                except Exception as db_err:
                    logger.error(f"Failed to record validation failure in DB: {db_err}")
                
                error_msg = "; ".join(val_res.get("errors", ["Unknown validation error"]))
                val_err = ValueError(f"Media validation failed: {error_msg}")
                val_err.validation_result = val_res
                raise val_err
            
            status = "Success"
        except Exception as e:
            if isinstance(e, ValueError) and "Media validation failed" in str(e):
                raise e
            context.add_log("Media Validation Agent", "Failed", start, time.time())
            raise ValueError(f"Media validation error: {str(e)}")
            
        context.add_log("Media Validation Agent", status, start, time.time())

        # 3. Automatic Media Type Routing
        # Orchestrator autonomously assigns media_type based strictly on detected media type.
        # This prevents manual forcing of image files to video pipeline or video files to image pipeline.
        detected_type = val_res.get("media_type")
        if not detected_type or detected_type not in ["image", "video"]:
            raise ValueError(f"Unsupported or unknown media type: {detected_type}")

        if context.media_type and context.media_type.lower() != detected_type:
            logger.warning(
                f"Requested media_type '{context.media_type}' does not match detected '{detected_type}'. "
                f"Overriding with detected type."
            )

        context.media_type = detected_type
        media_type = detected_type

        # Initialize database analysis record with verified detected media type
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

        # 4. Route to specialised analysis agent
        agent_name = "Image Agent" if media_type == "image" else "Video Agent"
        
        start = time.time()
        agent_res = None
        try:
            if media_type == 'image':
                if context.selected_providers is not None:
                    agent_res = self.image_agent.analyze(file_path, selected_providers=context.selected_providers)
                else:
                    agent_res = self.image_agent.analyze(file_path)
                context.image_agent_result = agent_res
            elif media_type == 'video':
                if context.selected_providers is not None:
                    agent_res = self.video_agent.analyze(file_path, selected_providers=context.selected_providers)
                else:
                    agent_res = self.video_agent.analyze(file_path)
                context.video_agent_result = agent_res
            else:
                raise ValueError(f"Unsupported media type: {context.media_type}")
            status = "Success"
            context.active_agents.append(agent_name)
        except Exception as e:
            logger.error(f"Primary Agent {agent_name} failed: {e}")
            context.add_log(agent_name, "Failed", start, time.time())
            context.failures.append({"agent": agent_name, "error": str(e)})
            raise RuntimeError(f"Primary specialist analysis failed: {str(e)}") from e
        
        context.add_log(agent_name, status, start, time.time())

        # 4. Metadata Agent
        start = time.time()
        metadata_res = None
        try:
            metadata_res = self.metadata_agent.analyze(file_path)
            context.metadata_agent_result = metadata_res
            context.active_agents.append("MetadataAgent")
            status = "Success"
        except Exception as e:
            logger.error(f"MetadataAgent failed: {e}")
            context.add_log("MetadataAgent", "Failed (Recovered)", start, time.time())
            context.failures.append({"agent": "MetadataAgent", "error": str(e)})
            metadata_res = {
                "status": "Failed",
                "metadata_present": False,
                "anomalies_detected": False,
                "software_tag": None,
                "suspicious_indicators": [],
                "summary": "Metadata analysis was unavailable due to an internal error."
            }
            context.metadata_agent_result = metadata_res
            status = "Success"
            
        if status == "Success":
            context.add_log("MetadataAgent", status, start, time.time())

        # 5. Confidence Evaluation Agent
        start = time.time()
        conf_res = None
        try:
            conf_res = self.confidence_agent.evaluate(agent_res)
            context.confidence_result = conf_res
            status = "Success"
        except Exception as e:
            logger.error(f"Confidence Evaluation Agent failed: {e}")
            context.add_log("Confidence Evaluation Agent", "Failed (Recovered)", start, time.time())
            context.failures.append({"agent": "Confidence Evaluation Agent", "error": str(e)})
            conf_res = {
                "confidence": agent_res.get("confidence_score", 50),
                "level": "Medium",
                "recommendation": "Unable to evaluate confidence tier."
            }
            context.confidence_result = conf_res
            status = "Success"
            
        if status == "Success":
            context.add_log("Confidence Evaluation Agent", status, start, time.time())

        # 6. DECISION NODE - Check if ArtifactConsistencyAgent is needed
        primary_conf_val = agent_res.get("confidence_score")
        primary_confidence = (float(primary_conf_val) / 100.0) if primary_conf_val is not None else 0.50
        primary_verdict = agent_res.get("verdict", "REAL")
        metadata_anomalies = metadata_res.get("anomalies_detected", False)
        
        # Conflict exists if primary says DEEPFAKE but metadata shows no anomalies, 
        # or primary says REAL but metadata shows anomalies (editing signatures present).
        has_conflict = (primary_verdict == "DEEPFAKE" and not metadata_anomalies) or (primary_verdict == "REAL" and metadata_anomalies)
        
        decision_reason = ""
        should_trigger_artifact = False
        
        if primary_verdict == "NOT_SUITABLE":
            should_trigger_artifact = False
            decision_reason = "ArtifactAgent skipped: Media is unsupported for facial forensic analysis (no face detected)."
        elif primary_confidence < PipelineConfig.AGENTIC_CONFIDENCE_THRESHOLD:
            should_trigger_artifact = True
            decision_reason = f"ArtifactAgent activated: primary confidence {round(primary_confidence, 2)} < threshold {PipelineConfig.AGENTIC_CONFIDENCE_THRESHOLD}"
        elif has_conflict:
            should_trigger_artifact = True
            decision_reason = f"ArtifactAgent activated: conflict detected (verdict={primary_verdict}, metadata_anomalies={metadata_anomalies})"
        else:
            decision_reason = f"ArtifactAgent skipped: primary confidence {round(primary_confidence, 2)} >= threshold {PipelineConfig.AGENTIC_CONFIDENCE_THRESHOLD} and metadata aligned"

        context.decision_history.append(decision_reason)
        logger.info(decision_reason)
        
        # Optionally execute Artifact Agent
        if should_trigger_artifact:
            start = time.time()
            try:
                artifact_res = self.artifact_agent.analyze(file_path, media_type=media_type)
                context.artifact_agent_result = artifact_res
                context.active_agents.append("ArtifactConsistencyAgent")
                status = "Success"
            except Exception as e:
                logger.error(f"ArtifactConsistencyAgent failed: {e}")
                context.add_log("ArtifactConsistencyAgent", "Failed (Recovered)", start, time.time())
                context.failures.append({"agent": "ArtifactConsistencyAgent", "error": str(e)})
                artifact_res = {
                    "status": "Failed",
                    "anomaly_score": None,
                    "verdict": "UNKNOWN",
                    "detected_artifacts": [],
                    "summary": "Artifact analysis failed (recovered)."
                }
                context.artifact_agent_result = artifact_res
                status = "Success"
                
            if status == "Success":
                context.add_log("ArtifactConsistencyAgent", status, start, time.time())

        # 7. Fusion Agent (Decision Maker)
        start = time.time()
        try:
            fusion_res = self.fusion_agent.fuse(context)
            context.fusion_result = fusion_res
            status = "Success"
        except Exception as e:
            logger.error(f"Fusion Agent failed: {e}")
            context.add_log("Fusion Agent", "Failed", start, time.time())
            raise e
        context.add_log("Fusion Agent", status, start, time.time())

        # 8. Explanation Agent
        start = time.time()
        try:
            # Merge additional agent findings into features for explanation enrichment
            explanation_features = dict(agent_res.get("features", {}))
            if metadata_res:
                explanation_features["metadata_anomaly"] = metadata_res.get("anomalies_detected", False)
                explanation_features["software_tag"] = metadata_res.get("software_tag")
            if context.artifact_agent_result:
                explanation_features["artifact_anomaly"] = context.artifact_agent_result.get("verdict") == "DEEPFAKE"
                explanation_features["anomaly_score"] = context.artifact_agent_result.get("anomaly_score", 0.0)

            canonical_v = fusion_res.get("canonical_verdict") or fusion_res.get("verdict") or fusion_res.get("primary_verdict")
            explanation_res = self.explanation_agent.explain(
                verdict=canonical_v,
                confidence_level=fusion_res.get("confidence_level", conf_res.get("level")),
                media_type=media_type,
                features=explanation_features,
                evidence=fusion_res.get("evidence", []),
                reason=fusion_res.get("reason"),
                user_facing_statement=fusion_res.get("user_facing_statement"),
            )
            context.explanation = explanation_res
            
            # Enrich fusion explanation to ensure single source of truth
            fusion_res["confidence_level"] = fusion_res.get("confidence_level", conf_res.get("level"))
            fusion_res["evidence"] = explanation_res.get("evidence", [])
            fusion_res["recommendation"] = explanation_res.get("recommendation", "N/A")
            fusion_res["user_facing_statement"] = explanation_res.get("user_facing_statement")
            status = "Success"
        except Exception as e:
            logger.error(f"Explanation Agent failed: {e}")
            context.add_log("Explanation Agent", "Failed", start, time.time())
            raise e
        context.add_log("Explanation Agent", status, start, time.time())

        # 7. PDF Report Generator
        start = time.time()
        try:
            report_path = ReportGenerator.generate_report(context.to_dict())
            context.report_path = report_path
            # Build clean report URL for the client
            fusion_res["report_url"] = f"/api/analysis/report/{context.analysis_id}.pdf"
            status = "Success"
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            context.add_log("PDF Report Generator", "Failed", start, time.time())
        else:
            context.add_log("PDF Report Generator", status, start, time.time())

        # 8. Save Analysis History (Consistent with Primary Verdict Single Source of Truth)
        start = time.time()
        try:
            user_id = context.user.get("id") if context.user else None
            user_name = context.user.get("name") if context.user else "Guest"
            saved_verdict = canonical_v
            conf_val = fusion_res.get("confidence")
            confidence = float(conf_val) if conf_val is not None else 0.0
            
            save_analysis(
                analysis_id=context.analysis_id,
                user_id=user_id,
                user_name=user_name,
                file_name=context.uploaded_file,
                media_type=context.media_type,
                prediction=saved_verdict,
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
