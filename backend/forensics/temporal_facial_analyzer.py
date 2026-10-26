import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from backend.utils.logger import setup_logger

logger = setup_logger("TemporalFacialAnalyzer")


class TemporalFaceTrack:
    """
    Maintains a continuous temporal history for a single tracked face across video frames.
    Tracks spatial bounding coordinates, neural probabilities, landmark textures,
    eye aperture / blinking patterns, oral cavity motion, and boundary seam flicker.
    """

    def __init__(self, track_id: str, initial_box: Tuple[int, int, int, int], frame_idx: int, timestamp: float):
        self.track_id = track_id
        self.created_at = timestamp
        self.last_seen_frame_idx = frame_idx
        self.last_seen_timestamp = timestamp
        
        self.boxes: List[Tuple[int, int, int, int]] = [initial_box]
        self.normalized_boxes: List[Tuple[float, float, float, float]] = []
        self.frame_indices: List[int] = [frame_idx]
        self.timestamps: List[float] = [timestamp]
        
        self.spatial_fake_probs: List[float] = []
        self.vit_fake_probs: List[float] = []
        self.xcep_fake_probs: List[float] = []
        self.eye_sharpness: List[float] = []
        self.eye_detected_counts: List[int] = []
        self.mouth_texture_vars: List[float] = []
        self.boundary_flickers: List[float] = []
        self.lighting_gradients: List[float] = []
        
        # In-memory cached crops (kept to a bounded count to prevent memory leaks)
        self.sampled_crops: List[np.ndarray] = []
        self.sampled_crop_indices: List[int] = []
        self._prev_boundary_patch: Optional[np.ndarray] = None

    @property
    def last_box(self) -> Tuple[int, int, int, int]:
        return self.boxes[-1] if self.boxes else (0, 0, 0, 0)

    @property
    def total_frames(self) -> int:
        return len(self.frame_indices)


class TemporalFacialAnalyzer:
    """
    Temporal Facial Forensics Engine for video stream and upload inspection.
    Implements multi-face tracking and four empirical temporal analyses:
    1. Eye aperture, blinking patterns, and ocular texture sharpness
    2. Mouth and facial movement, oral texture variance, and localized warping
    3. Face-swap blending boundary seams and inter-frame edge flicker
    4. Bilateral facial illumination and reflection consistency
    """

    def __init__(self):
        # Lazy load Haar cascades for fast startup
        self._face_cascade = None
        self._eye_cascade = None
        self.tracks: Dict[str, TemporalFaceTrack] = {}
        self._next_track_idx = 1

    @property
    def face_cascade(self):
        if self._face_cascade is None:
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(p)
        return self._face_cascade

    @property
    def eye_cascade(self):
        if self._eye_cascade is None:
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                p = cv2.data.haarcascades + "haarcascade_eye.xml"
                self._eye_cascade = cv2.CascadeClassifier(p)
        return self._eye_cascade

    def reset(self):
        """Resets all tracking states for a fresh video analysis or monitoring session."""
        self.tracks.clear()
        self._next_track_idx = 1

    def match_track(
        self,
        box: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        max_dist_threshold: float = 0.28,
        min_iou_threshold: float = 0.15,
    ) -> Optional[TemporalFaceTrack]:
        """
        Associates an incoming face box with an existing active face track using
        normalized centroid distance and bounding IoU.
        """
        bx, by, bw, bh = box
        bcx, bcy = bx + bw / 2.0, by + bh / 2.0

        best_track = None
        best_dist = float("inf")

        for track in self.tracks.values():
            lx, ly, lw, lh = track.last_box
            lcx, lcy = lx + lw / 2.0, ly + lh / 2.0

            norm_dist = np.hypot((bcx - lcx) / max(1, img_w), (bcy - lcy) / max(1, img_h))

            # Calculate Intersection over Union
            xA = max(bx, lx)
            yA = max(by, ly)
            xB = min(bx + bw, lx + lw)
            yB = min(by + bh, ly + lh)
            inter = max(0, xB - xA) * max(0, yB - yA)
            union = (bw * bh) + (lw * lh) - inter
            iou_val = inter / float(union) if union > 0 else 0.0

            if (iou_val >= min_iou_threshold or norm_dist <= max_dist_threshold) and norm_dist < best_dist:
                best_dist = norm_dist
                best_track = track

        return best_track

    def update_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        detected_faces: Optional[List[Tuple[int, int, int, int]]] = None,
        max_tracks: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Updates face tracks and extracts empirical temporal signals for each tracked face in the frame.
        Uses global cost-matrix association to ensure persistent identity tracking across frames.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        img_h, img_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        if detected_faces is None:
            if self.face_cascade is not None:
                detected_faces = [
                    (int(x), int(y), int(w), int(h))
                    for (x, y, w, h) in self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(24, 24))
                ]
            else:
                detected_faces = []

        # Filter out tiny spurious detections and sort by area descending
        valid_boxes = [b for b in detected_faces if b[2] >= 20 and b[3] >= 20]
        valid_boxes = sorted(valid_boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_tracks]

        # Greedy Cost-Matrix Association
        active_track_list = list(self.tracks.values())
        assignments: Dict[int, TemporalFaceTrack] = {} # box_idx -> track
        used_track_ids = set()

        if active_track_list and valid_boxes:
            candidates = []
            for b_idx, box in enumerate(valid_boxes):
                bx, by, bw, bh = box
                bcx, bcy = bx + bw / 2.0, by + bh / 2.0
                for track in active_track_list:
                    lx, ly, lw, lh = track.last_box
                    lcx, lcy = lx + lw / 2.0, ly + lh / 2.0
                    norm_dist = np.hypot((bcx - lcx) / max(1, img_w), (bcy - lcy) / max(1, img_h))

                    xA = max(bx, lx)
                    yA = max(by, ly)
                    xB = min(bx + bw, lx + lw)
                    yB = min(by + bh, ly + lh)
                    inter = max(0, xB - xA) * max(0, yB - yA)
                    union = (bw * bh) + (lw * lh) - inter
                    iou_val = inter / float(union) if union > 0 else 0.0

                    if iou_val >= 0.15 or norm_dist <= 0.35:
                        cost = norm_dist - (0.2 * iou_val)
                        candidates.append((cost, b_idx, track))

            candidates.sort(key=lambda c: c[0])
            for cost, b_idx, track in candidates:
                if b_idx not in assignments and track.track_id not in used_track_ids:
                    assignments[b_idx] = track
                    used_track_ids.add(track.track_id)

        frame_tracked_results = []
        for b_idx, box in enumerate(valid_boxes):
            bx, by, bw, bh = box
            track = assignments.get(b_idx)

            if track is None:
                track_id = f"T{self._next_track_idx}"
                self._next_track_idx += 1
                track = TemporalFaceTrack(track_id, box, frame_idx, timestamp)
                self.tracks[track_id] = track
            else:
                track.boxes.append(box)
                track.frame_indices.append(frame_idx)
                track.timestamps.append(timestamp)
                track.last_seen_frame_idx = frame_idx
                track.last_seen_timestamp = timestamp


            norm_box = (
                round(bx / img_w, 4),
                round(by / img_h, 4),
                round(bw / img_w, 4),
                round(bh / img_h, 4),
            )
            track.normalized_boxes.append(norm_box)

            # Crop face region
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(img_w, bx + bw), min(img_h, by + bh)
            face_roi_gray = gray[y1:y2, x1:x2]
            face_roi_bgr = frame[y1:y2, x1:x2]

            if face_roi_gray.size == 0 or face_roi_gray.shape[0] < 16 or face_roi_gray.shape[1] < 16:
                continue

            fh, fw = face_roi_gray.shape[:2]

            # Cache up to 10 sample crops evenly across frames
            if len(track.sampled_crops) < 10 or (frame_idx - track.sampled_crop_indices[-1] >= 5):
                track.sampled_crops.append(face_roi_bgr.copy())
                track.sampled_crop_indices.append(frame_idx)

            # -------------------------------------------------------------
            # 1. Eye Aperture & Texture Sharpness
            # -------------------------------------------------------------
            eye_y1 = int(0.18 * fh)
            eye_y2 = int(0.44 * fh)
            eye_x1 = int(0.10 * fw)
            eye_x2 = int(0.90 * fw)
            eye_roi = face_roi_gray[eye_y1:eye_y2, eye_x1:eye_x2]

            num_eyes_detected = 0
            eye_sobel_val = 0.0
            if eye_roi.size > 0 and eye_roi.shape[0] > 6 and eye_roi.shape[1] > 6:
                if self.eye_cascade is not None:
                    try:
                        detected_eyes = self.eye_cascade.detectMultiScale(eye_roi, 1.1, 2, minSize=(10, 10))
                        num_eyes_detected = len(detected_eyes)
                    except Exception:
                        pass
                # Vertical Sobel gradient measures eyelid/crease sharpness and aperture transitions
                sobel_y = cv2.Sobel(eye_roi, cv2.CV_64F, 0, 1, ksize=3)
                eye_sobel_val = float(np.mean(np.abs(sobel_y)))

            track.eye_detected_counts.append(num_eyes_detected)
            track.eye_sharpness.append(round(eye_sobel_val, 2))

            # -------------------------------------------------------------
            # 2. Mouth Texture Variance & Warping
            # -------------------------------------------------------------
            mouth_y1 = int(0.64 * fh)
            mouth_y2 = int(0.92 * fh)
            mouth_x1 = int(0.20 * fw)
            mouth_x2 = int(0.80 * fw)
            mouth_roi = face_roi_gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

            mouth_var = 0.0
            if mouth_roi.size > 0 and mouth_roi.shape[0] > 6 and mouth_roi.shape[1] > 6:
                mouth_var = float(cv2.Laplacian(mouth_roi, cv2.CV_64F).var())

            track.mouth_texture_vars.append(round(mouth_var, 2))

            # -------------------------------------------------------------
            # 3. Boundary Blending Seam & Temporal Edge Flicker
            # -------------------------------------------------------------
            margin = max(4, int(bw * 0.10))
            y_out_start = max(0, by - margin)
            y_out_end = min(img_h, by + bh + margin)
            x_out_start = max(0, bx - margin)
            x_out_end = min(img_w, bx + bw + margin)
            border_patch = gray[y_out_start:y_out_end, x_out_start:x_out_end]

            rim_flicker = 0.0
            if border_patch.size > 0 and border_patch.shape[0] > 10 and border_patch.shape[1] > 10:
                border_patch_resized = cv2.resize(border_patch, (96, 96))
                if track._prev_boundary_patch is not None:
                    diff = cv2.absdiff(border_patch_resized, track._prev_boundary_patch)
                    # Exclude the inner 60% of the face, isolating the outer boundary border seam
                    rim_mask = np.ones((96, 96), dtype=bool)
                    rim_mask[20:76, 20:76] = False
                    rim_flicker = float(np.mean(diff[rim_mask]))
                track._prev_boundary_patch = border_patch_resized

            track.boundary_flickers.append(round(rim_flicker, 2))

            # -------------------------------------------------------------
            # 4. Bilateral Cheek Illumination Gradient
            # -------------------------------------------------------------
            cheek_y1 = int(0.35 * fh)
            cheek_y2 = int(0.65 * fh)
            left_cheek = face_roi_gray[cheek_y1:cheek_y2, int(0.12 * fw):int(0.42 * fw)]
            right_cheek = face_roi_gray[cheek_y1:cheek_y2, int(0.58 * fw):int(0.88 * fw)]

            illum_grad = 0.0
            if left_cheek.size > 0 and right_cheek.size > 0:
                l_mean = float(np.mean(left_cheek))
                r_mean = float(np.mean(right_cheek))
                denom = max(1.0, (l_mean + r_mean) / 2.0)
                illum_grad = abs(l_mean - r_mean) / denom

            track.lighting_gradients.append(round(illum_grad, 3))

            frame_tracked_results.append({
                "track_id": track.track_id,
                "box": box,
                "box_normalized": norm_box,
                "total_frames_tracked": track.total_frames,
                "eye_sharpness": eye_sobel_val,
                "mouth_texture": mouth_var,
                "boundary_flicker": rim_flicker,
                "lighting_gradient": illum_grad,
            })

        return frame_tracked_results

    def compute_track_forensics(self, track: TemporalFaceTrack) -> Dict[str, Any]:
        """
        Synthesizes the complete empirical temporal forensic profile for a single tracked face.
        """
        n = track.total_frames
        if n == 0:
            return {}

        mouth_vars = track.mouth_texture_vars or [50.0]
        mean_mouth_texture = float(np.mean(mouth_vars))
        min_mouth_texture = float(np.min(mouth_vars))

        eye_sharp = track.eye_sharpness or [40.0]
        mean_eye_sharpness = float(np.mean(eye_sharp))
        min_eye_sharpness = float(np.min(eye_sharp))

        flickers = [f for f in track.boundary_flickers if f > 0]
        mean_boundary_flicker = float(np.mean(flickers)) if flickers else 0.0
        max_boundary_flicker = float(np.max(flickers)) if flickers else 0.0

        lighting = track.lighting_gradients or [0.10]
        mean_lighting_gradient = float(np.mean(lighting))

        spatial_scores = track.spatial_fake_probs or [0.10]
        mean_spatial_score = float(np.mean(spatial_scores))
        max_spatial_score = float(np.max(spatial_scores))
        score_variance = float(np.std(spatial_scores))

        vit_scores = track.vit_fake_probs or []
        mean_vit_score = round(float(np.mean(vit_scores)), 3) if vit_scores else None
        max_vit_score = round(float(np.max(vit_scores)), 3) if vit_scores else None
        is_model_conflict = bool(vit_scores and abs(float(np.mean(vit_scores)) - mean_spatial_score) >= 0.40)

        xcep_scores = track.xcep_fake_probs or []
        mean_xcep_score = round(float(np.mean(xcep_scores)), 3) if xcep_scores else None
        max_xcep_score = round(float(np.max(xcep_scores)), 3) if xcep_scores else None

        # -------------------------------------------------------------
        # Empirical Indicators of Manipulation
        # -------------------------------------------------------------
        # DeepFakes & Face Swaps exhibit:
        # a) Boundary Seam Jitter: mean_boundary_flicker > 32.0
        # b) Oral Cavity Smoothing: mean_mouth_texture < 25.0
        # c) Loss of Fine Ocular Sharpness: mean_eye_sharpness < 28.0
        # d) Persistent or elevated neural classification scores (mean > 0.30 or max > 0.45)
        is_boundary_anomalous = bool(mean_boundary_flicker > 34.0 or max_boundary_flicker > 48.0)
        is_mouth_smoothed = bool(mean_mouth_texture < 24.0 and min_mouth_texture < 18.0)
        is_eye_smoothed = bool(mean_eye_sharpness < 25.0)
        is_lighting_asymmetric = bool(mean_lighting_gradient > 0.20)
        is_neural_suspicious = bool(mean_spatial_score >= 0.32 or max_spatial_score >= 0.48)

        # Count corroborated empirical signals
        anomaly_count = sum([
            is_boundary_anomalous,
            is_mouth_smoothed,
            is_eye_smoothed,
            is_lighting_asymmetric,
            is_neural_suspicious,
        ])

        # Synthesize temporal manipulation score [0.0, 1.0]
        # Weights: Neural Spatial (35%), Boundary Seam Flicker (25%), Oral Texture Smoothing (20%),
        # Ocular Sharpness (10%), Lighting Asymmetry (10%)
        flicker_factor = min(1.0, mean_boundary_flicker / 45.0) if is_boundary_anomalous else 0.20
        mouth_factor = max(0.0, 1.0 - (mean_mouth_texture / 80.0)) if is_mouth_smoothed else 0.15
        eye_factor = max(0.0, 1.0 - (mean_eye_sharpness / 60.0)) if is_eye_smoothed else 0.15
        light_factor = min(1.0, mean_lighting_gradient / 0.25)

        temporal_manipulation_score = float(
            0.35 * mean_spatial_score
            + 0.25 * flicker_factor
            + 0.20 * mouth_factor
            + 0.10 * eye_factor
            + 0.10 * light_factor
        )

        return {
            "track_id": track.track_id,
            "total_frames_analyzed": n,
            "mean_spatial_score": round(mean_spatial_score, 3),
            "max_spatial_score": round(max_spatial_score, 3),
            "mean_vit_score": mean_vit_score,
            "max_vit_score": max_vit_score,
            "mean_xcep_score": mean_xcep_score,
            "max_xcep_score": max_xcep_score,
            "is_model_conflict": is_model_conflict,
            "score_variance": round(score_variance, 3),
            "mean_boundary_flicker": round(mean_boundary_flicker, 2),
            "max_boundary_flicker": round(max_boundary_flicker, 2),
            "mean_mouth_texture": round(mean_mouth_texture, 1),
            "mean_eye_sharpness": round(mean_eye_sharpness, 1),
            "mean_lighting_gradient": round(mean_lighting_gradient, 3),
            "temporal_manipulation_score": round(temporal_manipulation_score, 3),
            "anomaly_count": anomaly_count,
            "is_boundary_anomalous": is_boundary_anomalous,
            "is_mouth_smoothed": is_mouth_smoothed,
            "is_eye_smoothed": is_eye_smoothed,
            "is_lighting_asymmetric": is_lighting_asymmetric,
            "is_neural_suspicious": is_neural_suspicious,
        }
