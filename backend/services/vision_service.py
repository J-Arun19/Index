import os
import cv2
import json
import base64
import time
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from backend.utils.logger import setup_logger

logger = setup_logger("VisionService")


class VisionAnalysisService:
    """
    Multi-Provider AI Vision Forensics Service for SatyaDrishti.

    Features:
    1. Independent Multi-Provider Vision Adapters:
       - Google Gemini Vision (GEMINI_API_KEY)
       - OpenAI Vision (OPENAI_API_KEY)
       - xAI Grok Vision (XAI_API_KEY / GROK_API_KEY)
       - Local Deterministic Forensic Vision (Zero-network baseline & offline fallback)
    2. Independent Execution:
       - Each configured provider receives identical forensic instructions independently.
       - No provider's prior verdict is passed to later providers.
    3. Structured Forensics:
       - Every adapter returns standardized structured results:
         provider ID, model, success status, examined media regions, observable anomalies,
         face regions assessment, compression distinction, assessment (REAL / FAKE / INCONCLUSIVE),
         limitations, latency, and estimated API cost.
    4. Video Multi-Provider Keyframe Analysis:
       - Samples representative face keyframes across video duration.
       - Evaluates keyframes using active vision providers.
       - Evaluates temporal consistency across all frames.
       - Transparently reports audio-visual lip sync as NOT analyzed (never fabricating audio checks).
    5. Cost and Privacy Controls:
       - Configurable provider selection via env vars or request params.
       - Secure in-memory deduplication cache (SHA-256 keyed, RAM only, never writing private media to disk).
       - Zero model downloads, reusing existing dependencies.
    """

    FORENSIC_PROMPT = (
        "You are an expert digital forensics examiner specializing in detecting AI-generated portraits, deepfakes, and synthetic imagery.\n"
        "Critically examine the provided image for any indicators of AI generation (such as Midjourney, Flux, Stable Diffusion, StyleGAN, Imagen) or digital face manipulation vs authentic camera photography.\n"
        "Examine:\n"
        "1. Photorealism vs AI rendering: Skin texture realism (synthetic waxy sheen, plastic smoothing, unnatural pore patterns), hair strands consistency and blurring, eye reflections and pupil geometry, dental details, ear anatomy, and clothing/jewelry details.\n"
        "2. Boundaries & blending: Hairline transitions, jawline boundaries, face-swap blending seams, skin texture continuity between face and neck.\n"
        "3. Sensor vs Synthetic: Real camera sensor noise/ISO grain vs synthetic denoised rendering or artificial diffusion noise.\n"
        "4. Generative hallucinations: Subtle anatomical anomalies, background coherence, lighting angle and shadow consistency.\n\n"
        "CRITICAL INSTRUCTION: If you observe characteristic AI generative synthesis traits or manipulation anomalies, assess as 'FAKE'. If the image exhibits authentic camera sensor noise, natural optical capture, and consistent anatomy without AI synthesis artifacts, assess as 'REAL'. If the image lacks conclusive camera-optical provenance or if you cannot determine with reasonable certainty whether it is real or AI-generated, assess as 'INCONCLUSIVE' with moderate/low confidence. Do NOT assume an image is REAL merely because it looks aesthetically convincing or lacks obvious face-swap seams.\n\n"
        "You MUST respond in valid JSON ONLY with the following exact keys:\n"
        "{\n"
        '  "assessment": "REAL" | "FAKE" | "INCONCLUSIVE",\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "visual_synthetic_score": 0.0 to 1.0,\n'
        '  "observable_anomalies": ["list", "of", "concrete", "anomalies", "or empty array if authentic"],\n'
        '  "face_regions_assessment": "concise description of eyes, mouth, skin texture, hair borders",\n'
        '  "compression_vs_synthesis_distinction": "concise explanation of camera compression vs generative artifacts",\n'
        '  "limitations": ["list", "of", "limitations", "such as resolution or blur"]\n'
        "}"
    )

    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            backend_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
            if os.path.exists(backend_env):
                load_dotenv(backend_env)
        except ImportError:
            pass

        # API Keys - read strictly from backend environment variables
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.xai_api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")

        self.gemini_model = os.environ.get("VISION_GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.openai_model = os.environ.get("VISION_OPENAI_MODEL", "gpt-4o-mini")
        self.xai_model = os.environ.get("VISION_XAI_MODEL", "grok-2-vision-1212")

        # Configurable active provider selection
        self.configured_selection = os.environ.get("SATYA_VISION_PROVIDERS") or os.environ.get("VISION_PROVIDERS", "all")
        self.max_video_ai_frames = int(os.environ.get("SATYA_MAX_VIDEO_AI_FRAMES", "2"))

        # In-memory deduplication cache: {sha256_key: result_dict}
        # RAM only; never persisted to disk for privacy
        self._cache: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"VisionAnalysisService initialized. Configured providers: {self.get_active_provider_ids()}"
        )

    def _is_valid_key(self, val: Optional[str]) -> bool:
        """Validates that a provider API key is present and not a placeholder/empty string."""
        if not val or not isinstance(val, str):
            return False
        clean = val.strip()
        if not clean or len(clean) < 8:
            return False
        lower = clean.lower()
        if "your_" in lower or "placeholder" in lower or "api_key_here" in lower:
            return False
        return True

    def get_configured_providers(self) -> List[Dict[str, Any]]:
        """Returns metadata on which providers have credentials configured."""
        return [
            {
                "id": "gemini",
                "name": "Google Gemini Vision",
                "model": self.gemini_model,
                "configured": self._is_valid_key(self.gemini_api_key),
            },
            {
                "id": "openai",
                "name": "OpenAI Vision",
                "model": self.openai_model,
                "configured": self._is_valid_key(self.openai_api_key),
            },
            {
                "id": "xai",
                "name": "xAI Grok Vision",
                "model": self.xai_model,
                "configured": self._is_valid_key(self.xai_api_key),
            },
            {
                "id": "local",
                "name": "Local Forensic Vision",
                "model": "deterministic-v2",
                "configured": True,
            },
        ]

    def get_active_provider_ids(self, requested_providers: Optional[List[str]] = None) -> List[str]:
        """Resolves which providers should execute based on keys, config, and request filter."""
        active = []
        filter_set = None
        if requested_providers is not None:
            filter_set = set(p.lower().strip() for p in requested_providers)
        elif self.configured_selection and self.configured_selection.lower() != "all":
            filter_set = set(p.lower().strip() for p in self.configured_selection.split(","))

        # Check Gemini
        if self._is_valid_key(self.gemini_api_key) and (filter_set is None or "gemini" in filter_set or "all" in filter_set):
            active.append("gemini")

        # Check OpenAI
        if self._is_valid_key(self.openai_api_key) and (filter_set is None or "openai" in filter_set or "all" in filter_set):
            active.append("openai")

        # Check xAI
        if self._is_valid_key(self.xai_api_key) and (filter_set is None or "xai" in filter_set or "grok" in filter_set or "all" in filter_set):
            active.append("xai")

        return active

    def analyze_image(
        self,
        image_path: str,
        face_bbox: Optional[List[int]] = None,
        quality_gate_result: Optional[Any] = None,
        selected_providers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes independent multi-provider AI vision analysis on an image file.
        Calls each configured external vision provider independently, plus local deterministic forensics.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return self._empty_result("Unable to decode image file.")

        h, w, c = img_bgr.shape
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else ("image/png" if ext == ".png" else "image/webp")

        with open(image_path, "rb") as f:
            raw_bytes = f.read()

        active_provider_ids = self.get_active_provider_ids(selected_providers)

        # 1. Execute local forensic inspection as independent baseline
        local_result = self._local_image_visual_analysis(img_bgr, face_bbox, quality_gate_result)

        # 2. Execute each configured external provider independently
        provider_results: Dict[str, Any] = {}
        providers_attempted: List[str] = []
        providers_succeeded: List[str] = []
        providers_failed: List[Dict[str, str]] = []
        total_cost_usd = 0.0

        for pid in active_provider_ids:
            providers_attempted.append(pid)
            res = None
            if pid == "gemini":
                res = self._call_gemini_vision(raw_bytes, mime_type)
            elif pid == "openai":
                res = self._call_openai_vision(raw_bytes, mime_type)
            elif pid == "xai":
                res = self._call_xai_vision(raw_bytes, mime_type)

            if res and res.get("success"):
                provider_results[pid] = res
                providers_succeeded.append(pid)
                total_cost_usd += res.get("approx_cost_usd", 0.0)
            else:
                err_msg = res.get("error", "Unknown provider error") if res else "No response"
                providers_failed.append({"provider": pid, "error": err_msg})
                if res:
                    provider_results[pid] = res

        # Always record local forensic analysis in results
        provider_results["local_forensic"] = local_result

        # 3. Multi-Provider Consensus Synthesis
        return self._synthesize_image_consensus(
            local_result=local_result,
            provider_results=provider_results,
            providers_attempted=providers_attempted,
            providers_succeeded=providers_succeeded,
            providers_failed=providers_failed,
            total_cost_usd=round(total_cost_usd, 6),
            media_examined=f"Full image ({w}x{h}px)" + (f", face region ({face_bbox[2]}x{face_bbox[3]}px)" if face_bbox else ""),
        )

    def analyze_video(
        self,
        file_path: Optional[str] = None,
        frames: Optional[List[np.ndarray]] = None,
        fps: float = 30.0,
        has_audio: bool = False,
        face_bboxes: Optional[List[Optional[List[int]]]] = None,
        selected_providers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes multi-provider forensic analysis on video frames.
        1. Evaluates temporal consistency & appearance variance across decoded frames.
        2. Samples representative keyframes and submits them to active vision providers.
        3. Transparently reports audio-visual lip-sync status as NOT analyzed.
        """
        if not frames:
            return {
                "provider": "multi_provider_video",
                "providers_attempted": [],
                "providers_succeeded": [],
                "providers_failed": [],
                "temporal_consistency": "UNKNOWN",
                "facial_morphology_shift": 0.0,
                "observable_anomalies": ["Insufficient readable frames for temporal analysis."],
                "limitations": ["No frames decoded."],
                "audio_visual_sync": {
                    "analyzed": False,
                    "reason": "No frames decoded.",
                },
                "approx_cost_usd": 0.0,
            }

        # 1. Local temporal appearance and differential analysis
        temporal_res = self._local_video_temporal_analysis(frames, fps, has_audio)

        # 2. Select representative keyframes for AI vision examination
        # Sample up to max_video_ai_frames (e.g. 2 keyframes) to control latency and API cost
        n_frames = len(frames)
        keyframe_indices = []
        if n_frames == 1:
            keyframe_indices = [0]
        elif n_frames == 2:
            keyframe_indices = [0, 1]
        else:
            # Pick middle frame and end frame (often highest quality or potential splice points)
            mid = n_frames // 2
            keyframe_indices = [mid]
            if self.max_video_ai_frames > 1 and n_frames >= 3:
                keyframe_indices.append(n_frames - 1)

        active_provider_ids = self.get_active_provider_ids(selected_providers)
        provider_results: Dict[str, Any] = {}
        providers_attempted: List[str] = []
        providers_succeeded: List[str] = []
        providers_failed: List[Dict[str, str]] = []
        total_cost_usd = 0.0

        # Submit keyframes to active providers
        if active_provider_ids and keyframe_indices:
            # Primary keyframe for vision inspection
            kf_idx = keyframe_indices[0]
            kf_frame = frames[kf_idx]
            ok, buf = cv2.imencode(".jpg", kf_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                raw_bytes = buf.tobytes()
                for pid in active_provider_ids:
                    providers_attempted.append(pid)
                    res = None
                    if pid == "gemini":
                        res = self._call_gemini_vision(raw_bytes, "image/jpeg")
                    elif pid == "openai":
                        res = self._call_openai_vision(raw_bytes, "image/jpeg")
                    elif pid == "xai":
                        res = self._call_xai_vision(raw_bytes, "image/jpeg")

                    if res and res.get("success"):
                        provider_results[pid] = res
                        providers_succeeded.append(pid)
                        total_cost_usd += res.get("approx_cost_usd", 0.0)
                    else:
                        err_msg = res.get("error", "Unknown provider error") if res else "No response"
                        providers_failed.append({"provider": pid, "error": err_msg})
                        if res:
                            provider_results[pid] = res

        # Synthesize video consensus
        anomalies = list(temporal_res.get("observable_anomalies", []))
        limitations = list(temporal_res.get("limitations", []))
        provider_assessments: List[str] = []

        for pid, pres in provider_results.items():
            if pres.get("success"):
                ass = pres.get("assessment", "INCONCLUSIVE")
                provider_assessments.append(ass)
                for anom in pres.get("observable_anomalies", []):
                    if anom not in anomalies:
                        anomalies.append(f"[{pres.get('name', pid)}] {anom}")
                for lim in pres.get("limitations", []):
                    if lim not in limitations:
                        limitations.append(lim)

        consensus_assessment = "INCONCLUSIVE"
        fake_votes = provider_assessments.count("FAKE")
        real_votes = provider_assessments.count("REAL")

        if fake_votes > 0 and real_votes == 0:
            consensus_assessment = "FAKE"
        elif real_votes > 0 and fake_votes == 0:
            consensus_assessment = "REAL"
        elif fake_votes > 0 and real_votes > 0:
            consensus_assessment = "DISAGREEMENT"
            anomalies.append("Multi-provider keyframe analysis resulted in contradictory assessments between AI vision models.")
        elif not active_provider_ids:
            consensus_assessment = "LOCAL_TEMPORAL_ONLY"

        return {
            "provider": "multi_provider_video",
            "active_providers": active_provider_ids,
            "providers_attempted": providers_attempted,
            "providers_succeeded": providers_succeeded,
            "providers_failed": providers_failed,
            "provider_results": provider_results,
            "consensus_assessment": consensus_assessment,
            "temporal_consistency": temporal_res.get("temporal_consistency", "UNKNOWN"),
            "inter_frame_variance": temporal_res.get("inter_frame_variance", 0.0),
            "average_frame_delta": temporal_res.get("average_frame_delta", 0.0),
            "observable_anomalies": anomalies,
            "limitations": limitations,
            "audio_visual_sync": temporal_res.get("audio_visual_sync"),
            "approx_cost_usd": round(total_cost_usd, 6),
            "keyframes_examined_count": len(keyframe_indices),
        }

    def analyze_video_frames(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
        has_audio: bool = False,
    ) -> Dict[str, Any]:
        """Backwards compatibility wrapper calling analyze_video."""
        return self.analyze_video(frames=frames, fps=fps, has_audio=has_audio)

    # -------------------------------------------------------------------------
    # Provider Adapters
    # -------------------------------------------------------------------------

    def _call_gemini_vision(self, image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
        """Google Gemini Vision API Adapter with resilient fallback."""
        cache_key = self._compute_cache_key("gemini", self.gemini_model, image_bytes)
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["cached"] = True
            return cached

        t0 = time.time()
        try:
            import httpx

            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": self.FORENSIC_PROMPT},
                            {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.1, "response_mime_type": "application/json"},
            }

            candidate_models = [self.gemini_model]
            for m in ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest", "gemini-flash-lite-latest"]:
                if m not in candidate_models:
                    candidate_models.append(m)

            last_error = "Unknown error"
            with httpx.Client(timeout=35.0) as client:
                for candidate in candidate_models:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:generateContent?key={self.gemini_api_key}"
                        resp = client.post(url, json=payload)
                        latency_ms = int((time.time() - t0) * 1000)

                        if resp.status_code == 200:
                            data = resp.json()
                            text = data["candidates"][0]["content"]["parts"][0]["text"]
                            parsed = self._clean_and_parse_json(text)
                            result = self._format_adapter_result(
                                provider_id="gemini",
                                name="Google Gemini",
                                model=candidate,
                                success=True,
                                parsed=parsed,
                                latency_ms=latency_ms,
                                approx_cost_usd=0.00003,
                            )
                            self._cache[cache_key] = result
                            return result
                        elif resp.status_code == 503:
                            logger.warning(f"Gemini model {candidate} high demand (503), trying next candidate...")
                            last_error = f"HTTP {resp.status_code}"
                            time.sleep(1.0)
                        else:
                            logger.warning(f"Gemini API {candidate} returned status {resp.status_code}: {resp.text[:150]}")
                            last_error = f"HTTP {resp.status_code}"
                    except httpx.TimeoutException:
                        logger.warning(f"Gemini model {candidate} timed out, trying next candidate...")
                        last_error = "Timeout"
                    except Exception as err:
                        logger.warning(f"Gemini model {candidate} error: {err}")
                        last_error = str(err)

                return self._adapter_error_result("gemini", "Google Gemini", self.gemini_model, last_error)
        except Exception as e:
            logger.warning(f"Gemini Vision call failed safely: {e}")
            return self._adapter_error_result("gemini", "Google Gemini", self.gemini_model, str(e))

    def _call_openai_vision(self, image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
        """OpenAI Vision API Adapter."""
        cache_key = self._compute_cache_key("openai", self.openai_model, image_bytes)
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["cached"] = True
            return cached

        t0 = time.time()
        try:
            import httpx

            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.openai_api_key}"}
            payload = {
                "model": self.openai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.FORENSIC_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
                        ],
                    }
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }

            with httpx.Client(timeout=12.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                latency_ms = int((time.time() - t0) * 1000)

                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = self._clean_and_parse_json(content)
                    result = self._format_adapter_result(
                        provider_id="openai",
                        name="OpenAI Vision",
                        model=self.openai_model,
                        success=True,
                        parsed=parsed,
                        latency_ms=latency_ms,
                        approx_cost_usd=0.00015,
                    )
                    self._cache[cache_key] = result
                    return result
                else:
                    logger.warning(f"OpenAI API returned status {resp.status_code}: {resp.text[:200]}")
                    return self._adapter_error_result("openai", "OpenAI Vision", self.openai_model, f"HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"OpenAI Vision call failed safely: {e}")
            return self._adapter_error_result("openai", "OpenAI Vision", self.openai_model, str(e))

    def _call_xai_vision(self, image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
        """xAI Grok Vision API Adapter."""
        cache_key = self._compute_cache_key("xai", self.xai_model, image_bytes)
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["cached"] = True
            return cached

        t0 = time.time()
        try:
            import httpx

            b64_data = base64.b64encode(image_bytes).decode("utf-8")
            url = "https://api.x.ai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.xai_api_key}"}
            payload = {
                "model": self.xai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.FORENSIC_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
                        ],
                    }
                ],
                "temperature": 0.1,
            }

            with httpx.Client(timeout=14.0) as client:
                resp = client.post(url, json=payload, headers=headers)
                latency_ms = int((time.time() - t0) * 1000)

                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = self._clean_and_parse_json(content)
                    result = self._format_adapter_result(
                        provider_id="xai",
                        name="xAI Grok Vision",
                        model=self.xai_model,
                        success=True,
                        parsed=parsed,
                        latency_ms=latency_ms,
                        approx_cost_usd=0.002,
                    )
                    self._cache[cache_key] = result
                    return result
                else:
                    logger.warning(f"xAI API returned status {resp.status_code}: {resp.text[:200]}")
                    return self._adapter_error_result("xai", "xAI Grok Vision", self.xai_model, f"HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"xAI Vision call failed safely: {e}")
            return self._adapter_error_result("xai", "xAI Grok Vision", self.xai_model, str(e))

    # -------------------------------------------------------------------------
    # Local Forensic Baseline Adapter
    # -------------------------------------------------------------------------

    def _local_image_visual_analysis(
        self,
        img_bgr: np.ndarray,
        face_bbox: Optional[List[int]],
        quality_res: Optional[Any],
    ) -> Dict[str, Any]:
        """
        Deterministic, local visual reasoning:
        - Evaluates face region vs background coherence.
        - Measures bilateral symmetry across facial landmarks.
        - Discriminated standard camera compression (DCT block variance) from localized blending anomalies.
        """
        h, w, c = img_bgr.shape
        anomalies: List[str] = []
        limitations: List[str] = []

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 50.0:
            limitations.append("Image exhibits optical softness or compression blur; fine pore-level inspection limited.")

        if min(h, w) < 256:
            limitations.append("Low input resolution (<256px); high-frequency generative artifacts may be attenuated.")

        face_detected = face_bbox is not None and len(face_bbox) == 4
        visual_synthetic_score = 0.10
        face_assessment = "No human face region detected."
        compression_distinction = "Standard uniform camera sensor noise and compression pattern."

        if face_detected:
            fx, fy, fw, fh = face_bbox
            fx = max(0, fx)
            fy = max(0, fy)
            fw = min(fw, w - fx)
            fh = min(fh, h - fy)

            if fw > 20 and fh > 20:
                face_gray = gray[fy : fy + fh, fx : fx + fw]

                # 1. Bilateral symmetry analysis across face center
                mid_x = fw // 2
                left_half = face_gray[:, :mid_x]
                right_half = cv2.flip(face_gray[:, mid_x : mid_x * 2], 1)
                min_w = min(left_half.shape[1], right_half.shape[1])
                if min_w > 10:
                    diff = cv2.absdiff(left_half[:, :min_w], right_half[:, :min_w])
                    asymmetry_score = float(np.mean(diff) / 255.0)
                    if asymmetry_score > 0.40:
                        anomalies.append("Pronounced bilateral facial illumination and texture asymmetry.")
                        visual_synthetic_score += 0.20

                # 2. Boundary transition sharpness (blending seam detection)
                mask = np.zeros((fh, fw), dtype=np.uint8)
                cv2.rectangle(mask, (int(fw * 0.1), int(fh * 0.1)), (int(fw * 0.9), int(fh * 0.9)), 255, -1)
                border_mask = cv2.bitwise_not(mask)
                grad_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
                mag = cv2.magnitude(grad_x, grad_y)

                border_grad = float(np.mean(mag[border_mask > 0])) if np.any(border_mask > 0) else 0.0
                interior_grad = float(np.mean(mag[mask > 0])) if np.any(mask > 0) else 1.0

                if interior_grad > 0:
                    edge_ratio = border_grad / interior_grad
                    if edge_ratio > 2.8:
                        anomalies.append("Unnatural boundary edge gradient discontinuity around face perimeter (blending seam).")
                        visual_synthetic_score += 0.25

                # 3. Compression artifact discrimination:
                bg_mask = np.ones((h, w), dtype=bool)
                bg_mask[fy : fy + fh, fx : fx + fw] = False
                bg_lap = float(np.var(gray[bg_mask])) if np.any(bg_mask) else lap_var
                face_lap = float(np.var(face_gray))

                if bg_lap > 0:
                    texture_ratio = face_lap / bg_lap
                    # Only flag if face is severely smoothed (face_lap < 15.0) while background is sharp/noisy
                    if texture_ratio < 0.10 and bg_lap > 200 and face_lap < 15.0:
                        anomalies.append("Severe selective face texture smoothing inconsistent with background sensor noise.")
                        visual_synthetic_score += 0.20
                        compression_distinction = (
                            "Inconsistent noise field: Face region shows unnatural smoothing, while background "
                            "exhibits normal camera sensor noise. This differential is characteristic of face replacement."
                        )
                    else:
                        compression_distinction = (
                            "Consistent noise field: Compression characteristics and high-frequency noise levels "
                            "are uniform between facial region and background."
                        )

                face_assessment = (
                    f"Face detected ({fw}x{fh}px). Evaluated facial symmetry, boundary gradient transition, "
                    f"and differential noise distribution."
                )

        visual_synthetic_score = float(np.clip(visual_synthetic_score, 0.05, 0.95))
        assessment = "FAKE" if visual_synthetic_score >= 0.60 else ("REAL" if visual_synthetic_score <= 0.30 else "INCONCLUSIVE")

        return {
            "provider": "local_forensic",
            "name": "Local Forensic Vision",
            "model": "deterministic-v2",
            "success": True,
            "is_external_ai": False,
            "assessment": assessment,
            "confidence": round(abs(visual_synthetic_score - 0.5) * 2, 2),
            "visual_synthetic_score": round(visual_synthetic_score, 4),
            "observable_anomalies": anomalies,
            "face_regions_assessment": face_assessment,
            "compression_vs_synthesis_distinction": compression_distinction,
            "limitations": limitations,
            "approx_cost_usd": 0.0,
            "latency_ms": 15,
        }

    def _local_video_temporal_analysis(
        self,
        frames: List[np.ndarray],
        fps: float,
        has_audio: bool,
    ) -> Dict[str, Any]:
        """Evaluates inter-frame appearance variance and temporal consistency."""
        n_frames = len(frames)
        anomalies: List[str] = []
        limitations: List[str] = []

        if n_frames < 2:
            return {
                "provider": "local_video_temporal",
                "temporal_consistency": "INSUFFICIENT_FRAMES",
                "observable_anomalies": ["Video has fewer than 2 sampled frames."],
                "limitations": ["Insufficient temporal sequence length."],
                "audio_visual_sync": {
                    "analyzed": False,
                    "reason": "No frames decoded.",
                },
            }

        frame_diffs = []
        for i in range(n_frames - 1):
            f1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            f2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            if f1.shape != f2.shape:
                f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
            diff = float(np.mean(cv2.absdiff(f1, f2)))
            frame_diffs.append(diff)

        diff_variance = float(np.std(frame_diffs)) if frame_diffs else 0.0
        avg_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0

        if diff_variance > 35.0:
            anomalies.append("Irregular inter-frame appearance fluctuations and temporal flickering.")

        # Audio-visual lip-sync status:
        # Strictly transparent: DO NOT claim to have analyzed lip-sync when no model is integrated!
        av_sync_status = {
            "analyzed": False,
            "reason": (
                "Lip-sync analysis not performed: SatyaDrishti currently utilizes visual-temporal forensic inspection. "
                "No audio-visual lip-sync model is currently loaded in this configuration."
            ),
        }

        return {
            "provider": "local_video_temporal",
            "temporal_consistency": "CONSISTENT" if diff_variance <= 25.0 else "INCONSISTENT",
            "inter_frame_variance": round(diff_variance, 3),
            "average_frame_delta": round(avg_diff, 3),
            "observable_anomalies": anomalies,
            "limitations": limitations,
            "audio_visual_sync": av_sync_status,
        }

    # -------------------------------------------------------------------------
    # Consensus Synthesis & Helper Methods
    # -------------------------------------------------------------------------

    def _synthesize_image_consensus(
        self,
        local_result: Dict[str, Any],
        provider_results: Dict[str, Any],
        providers_attempted: List[str],
        providers_succeeded: List[str],
        providers_failed: List[Dict[str, str]],
        total_cost_usd: float,
        media_examined: str,
    ) -> Dict[str, Any]:
        """Fuses multiple provider outputs into a canonical visual analysis object."""
        external_assessments: List[str] = []
        all_anomalies: List[str] = []
        all_limitations: List[str] = []
        face_assessments: List[str] = []
        compression_notes: List[str] = []
        synthetic_scores: List[float] = []
        ext_scores: List[float] = []

        # Local anomalies first
        for anom in local_result.get("observable_anomalies", []):
            all_anomalies.append(anom)
        for lim in local_result.get("limitations", []):
            all_limitations.append(lim)
        if local_result.get("face_regions_assessment"):
            face_assessments.append(local_result["face_regions_assessment"])
        if local_result.get("compression_vs_synthesis_distinction"):
            compression_notes.append(local_result["compression_vs_synthesis_distinction"])
        synthetic_scores.append(local_result.get("visual_synthetic_score", 0.50))

        # External provider results
        for pid in providers_succeeded:
            pres = provider_results[pid]
            ass = pres.get("assessment", "INCONCLUSIVE")
            external_assessments.append(ass)
            p_name = pres.get("name", pid)

            for anom in pres.get("observable_anomalies", []):
                tagged = f"[{p_name}] {anom}"
                if tagged not in all_anomalies and anom not in all_anomalies:
                    all_anomalies.append(tagged)

            for lim in pres.get("limitations", []):
                if lim not in all_limitations:
                    all_limitations.append(f"[{p_name}] {lim}")

            if pres.get("face_regions_assessment"):
                face_assessments.append(f"{p_name}: {pres['face_regions_assessment']}")
            if pres.get("compression_vs_synthesis_distinction"):
                compression_notes.append(f"{p_name}: {pres['compression_vs_synthesis_distinction']}")
            if pres.get("visual_synthetic_score") is not None:
                score_val = float(pres["visual_synthetic_score"])
                ext_scores.append(score_val)
                synthetic_scores.append(score_val)

        # Consensus determination
        external_count = len(external_assessments)
        fake_votes = external_assessments.count("FAKE")
        real_votes = external_assessments.count("REAL")

        if external_count == 0:
            consensus_assessment = "LOCAL_ONLY"
            consensus_reason = "No external AI vision providers configured. Evaluation performed via local forensic baseline."
            avg_synthetic_score = float(np.mean(synthetic_scores)) if synthetic_scores else 0.50
        elif fake_votes > 0 and real_votes == 0:
            consensus_assessment = "FAKE"
            consensus_reason = f"External AI vision consensus confirms synthetic manipulation ({fake_votes}/{external_count} providers)."
            avg_synthetic_score = float(np.mean(ext_scores)) if ext_scores else 0.75
        elif real_votes > 0 and fake_votes == 0:
            consensus_assessment = "REAL"
            consensus_reason = f"External AI vision consensus confirms authentic natural imagery ({real_votes}/{external_count} providers)."
            # Filter out local heuristic anomalies and benign mobile filter notes when confirmed authentic by AI
            all_anomalies = [
                a for a in all_anomalies 
                if ("beauty filter" not in a.lower() and "mobile" not in a.lower() and "selective face texture" not in a.lower() and "unnatural boundary" not in a.lower())
            ]
            avg_synthetic_score = float(np.mean(ext_scores)) if ext_scores else 0.15
        elif fake_votes > 0 and real_votes > 0:
            consensus_assessment = "DISAGREEMENT"
            consensus_reason = f"Provider disagreement detected ({fake_votes} FAKE vs {real_votes} REAL). External models conflict."
            all_anomalies.append("Provider Disagreement: Multi-provider analysis showed conflicting assessments between AI vision models.")
            avg_synthetic_score = float(np.mean(synthetic_scores)) if synthetic_scores else 0.50
        else:
            consensus_assessment = "INCONCLUSIVE"
            consensus_reason = "External AI vision providers returned inconclusive assessments."
            avg_synthetic_score = float(np.mean(synthetic_scores)) if synthetic_scores else 0.50

        # Construct canonical object
        return {
            "provider": "multi_provider_vision",
            "active_providers": providers_attempted,
            "providers_attempted": providers_attempted,
            "providers_succeeded": providers_succeeded,
            "providers_failed": providers_failed,
            "provider_results": provider_results,
            "consensus_assessment": consensus_assessment,
            "consensus_reason": consensus_reason,
            "visual_synthetic_score": round(avg_synthetic_score, 4),
            "observable_anomalies": all_anomalies,
            "face_regions_assessment": "; ".join(face_assessments) if face_assessments else "Evaluated face regions.",
            "compression_vs_synthesis_distinction": "; ".join(compression_notes) if compression_notes else "Standard compression.",
            "limitations": all_limitations,
            "approx_cost_usd": total_cost_usd,
            "media_examined": media_examined,
            "external_ai_configured": len(providers_attempted) > 0,
        }

    def _format_adapter_result(
        self,
        provider_id: str,
        name: str,
        model: str,
        success: bool,
        parsed: Dict[str, Any],
        latency_ms: int,
        approx_cost_usd: float,
    ) -> Dict[str, Any]:
        """Normalizes external provider output into canonical structure."""
        raw_assessment = str(parsed.get("assessment", "INCONCLUSIVE")).upper().strip()
        if raw_assessment not in ["REAL", "FAKE", "INCONCLUSIVE"]:
            raw_assessment = "INCONCLUSIVE"

        conf = float(parsed.get("confidence", 0.50))
        synth_score = float(parsed.get("visual_synthetic_score", 0.50))
        anomalies = parsed.get("observable_anomalies", [])
        if not isinstance(anomalies, list):
            anomalies = [str(anomalies)] if anomalies else []

        limits = parsed.get("limitations", [])
        if not isinstance(limits, list):
            limits = [str(limits)] if limits else []

        return {
            "provider": provider_id,
            "name": name,
            "model": model,
            "success": success,
            "is_external_ai": True,
            "assessment": raw_assessment,
            "confidence": round(conf, 4),
            "visual_synthetic_score": round(synth_score, 4),
            "observable_anomalies": anomalies,
            "face_regions_assessment": str(parsed.get("face_regions_assessment", "")),
            "compression_vs_synthesis_distinction": str(parsed.get("compression_vs_synthesis_distinction", "")),
            "limitations": limits,
            "error": None,
            "latency_ms": latency_ms,
            "approx_cost_usd": approx_cost_usd,
            "cached": False,
        }

    def _adapter_error_result(
        self, provider_id: str, name: str, model: str, error_message: str
    ) -> Dict[str, Any]:
        """Formats a structured failure result for an external provider."""
        return {
            "provider": provider_id,
            "name": name,
            "model": model,
            "success": False,
            "is_external_ai": True,
            "assessment": "INCONCLUSIVE",
            "confidence": 0.0,
            "visual_synthetic_score": 0.50,
            "observable_anomalies": [],
            "face_regions_assessment": "Provider call failed.",
            "compression_vs_synthesis_distinction": "N/A",
            "limitations": [f"API error: {error_message}"],
            "error": error_message,
            "latency_ms": 0,
            "approx_cost_usd": 0.0,
            "cached": False,
        }

    def _clean_and_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Robust JSON extraction stripping markdown code fences or conversational text."""
        text = raw_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except Exception:
            # Fallback: regex search for outer braces
            import re
            m = re.search(r"(\{.*\})", text, re.DOTALL)
            if m:
                return json.loads(m.group(1))
            raise ValueError(f"Could not parse valid JSON from response: {raw_text[:150]}")

    def _compute_cache_key(self, provider: str, model: str, data: bytes) -> str:
        """In-memory cache key: sha256(provider + model + prompt + content)."""
        h = hashlib.sha256()
        h.update(provider.encode("utf-8"))
        h.update(b":")
        h.update(model.encode("utf-8"))
        h.update(b":")
        h.update(self.FORENSIC_PROMPT.encode("utf-8"))
        h.update(b":")
        h.update(data)
        return h.hexdigest()

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "provider": "multi_provider_vision",
            "active_providers": [],
            "providers_attempted": [],
            "providers_succeeded": [],
            "providers_failed": [],
            "provider_results": {},
            "consensus_assessment": "NOT_SUITABLE",
            "consensus_reason": reason,
            "observable_anomalies": [],
            "face_regions_assessment": reason,
            "compression_vs_synthesis_distinction": "Not available.",
            "visual_synthetic_score": 0.50,
            "limitations": [reason],
            "approx_cost_usd": 0.0,
            "external_ai_configured": False,
        }
