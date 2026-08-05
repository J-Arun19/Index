import os
from backend.config import Config
from backend.agents.image_agent import ImageAgent
from backend.agents.video_agent import VideoAgent
from backend.agents.fusion_agent import FusionAgent

class Orchestrator:
    def __init__(self):
        self.image_agent = ImageAgent()
        self.video_agent = VideoAgent()
        self.fusion_agent = FusionAgent()

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

    def run_orchestration(self, file_id, media_type):
        """
        Coordinates the workflow:
        1. Find file path by ID.
        2. Route based on media type.
        3. Execute Agent.
        4. Forward to Fusion Agent.
        5. Return JSON-compatible trace dictionary.
        """
        # 1. Locate file
        file_path = self.locate_file(file_id)
        if not file_path:
            raise FileNotFoundError(f"No file found with ID: {file_id}")
            
        # 2. Determine agent and route
        # Ensure media type is lowercase for matching
        mtype = media_type.lower()
        if mtype == 'image':
            agent_name = "ImageAgent"
            agent_output = self.image_agent.analyze(file_path)
        elif mtype == 'video':
            agent_name = "VideoAgent"
            agent_output = self.video_agent.analyze(file_path)
        else:
            raise ValueError(f"Unsupported media type: {media_type}. Must be 'image' or 'video'.")
            
        # 3. Fuse outputs
        fusion_output = self.fusion_agent.fuse(agent_name, agent_output)
        
        # 4. Assemble orchestrator response
        return {
            "file_id": file_id,
            "media_type": mtype,
            "agent_executed": agent_name,
            "analysis_status": "completed",
            "next_processing_stage": "Result",
            "agent_output": agent_output,
            "fusion_result": fusion_output
        }
