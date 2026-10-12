import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from backend.utils.logger import setup_logger

logger = setup_logger("ForensicAnalyzers")


class FFTSpectralAnalyzer:
    """
    Analyzes frequency domain characteristics using 2D Fast Fourier Transform.
    Detects periodic grid artifacts and anomalous spectral roll-off typical of
    generative models (GAN checkerboard upsampling artifacts, diffusion noise).
    """

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            image: BGR or Grayscale numpy image array.
        Returns:
            Dict containing frequency metrics, anomaly score, and factual description.
        """
        if image is None or image.size == 0:
            return {
                "fft_score": 0.0,
                "high_freq_ratio": 0.0,
                "anomaly_detected": False,
                "description": "Empty image provided to frequency analyzer.",
            }

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        # Normalize to float
        gray_f = gray.astype(np.float32) / 255.0

        # Apply 2D FFT and center the DC component
        f_transform = np.fft.fft2(gray_f)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log1p(np.abs(f_shift))

        # Center coordinates
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)

        if max_dist == 0:
            return {
                "fft_score": 0.0,
                "high_freq_ratio": 0.0,
                "anomaly_detected": False,
                "description": "Zero image dimensions in FFT analysis.",
            }

        normalized_dist = dist_from_center / max_dist

        # Measure energy in low, mid, and high frequency bands
        low_band = magnitude_spectrum[normalized_dist < 0.2]
        mid_band = magnitude_spectrum[(normalized_dist >= 0.2) & (normalized_dist < 0.6)]
        high_band = magnitude_spectrum[normalized_dist >= 0.6]

        low_energy = float(np.mean(low_band)) if low_band.size > 0 else 1.0
        mid_energy = float(np.mean(mid_band)) if mid_band.size > 0 else 1.0
        high_energy = float(np.mean(high_band)) if high_band.size > 0 else 0.0

        # In natural images, power decays as 1/f^alpha (steep fall-off).
        # Generative models often have elevated high-frequency energy or periodic peaks.
        high_to_mid_ratio = high_energy / max(1e-5, mid_energy)

        # Measure azimuthal standard deviation (rotational asymmetry in high frequencies)
        # Periodic grid artifacts create directional spikes in the spectrum.
        quadrants = [
            magnitude_spectrum[:cy, :cx],
            magnitude_spectrum[:cy, cx:],
            magnitude_spectrum[cy:, :cx],
            magnitude_spectrum[cy:, cx:],
        ]
        quadrant_means = [float(np.mean(q)) for q in quadrants if q.size > 0]
        quadrant_std = float(np.std(quadrant_means)) if len(quadrant_means) > 1 else 0.0

        # Compute combined spectral anomaly score between 0.0 and 1.0
        # Natural images typically have high_to_mid_ratio between 0.15 and 0.45
        anomaly_score = 0.10
        if high_to_mid_ratio > 0.60:
            anomaly_score += min(0.40, (high_to_mid_ratio - 0.60) * 1.2)
        if quadrant_std > 0.15:
            anomaly_score += min(0.35, quadrant_std * 1.5)

        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))
        anomaly_detected = bool(anomaly_score > 0.45)

        description = (
            f"Frequency domain: High-to-mid ratio {high_to_mid_ratio:.2f}, "
            f"spectral asymmetry {quadrant_std:.3f}. "
        )
        if anomaly_detected:
            description += "Anomalous high-frequency spectral distribution detected."
        else:
            description += "Natural 1/f spectral energy decay observed."

        return {
            "fft_score": round(anomaly_score, 4),
            "high_freq_ratio": round(high_to_mid_ratio, 4),
            "quadrant_asymmetry": round(quadrant_std, 4),
            "anomaly_detected": anomaly_detected,
            "description": description,
        }


class ErrorLevelAnalyzer:
    """
    Error Level Analysis (ELA) identifies differences in compression levels.
    When a face is spliced or altered, the modified region often exhibits a
    different compression response compared to the rest of the image.
    """

    def analyze(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Args:
            image: BGR numpy image array.
            face_bbox: Optional (x, y, w, h) bounding box of detected face.
        Returns:
            Dict containing ELA metrics, anomaly score, and description.
        """
        if image is None or image.size == 0:
            return {
                "ela_score": 0.0,
                "local_discrepancy": 0.0,
                "anomaly_detected": False,
                "description": "Empty image provided to ELA analyzer.",
            }

        # Resave at fixed 90% JPEG quality in memory
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        success, encoded = cv2.imencode(".jpg", image, encode_params)
        if not success:
            return {
                "ela_score": 0.0,
                "local_discrepancy": 0.0,
                "anomaly_detected": False,
                "description": "JPEG encoding failed during ELA.",
            }

        resaved = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if resaved is None:
            return {
                "ela_score": 0.0,
                "local_discrepancy": 0.0,
                "anomaly_detected": False,
                "description": "JPEG decoding failed during ELA.",
            }

        # Absolute difference image
        diff = cv2.absdiff(image, resaved).astype(np.float32)
        diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

        global_mean = float(np.mean(diff_gray))
        global_std = float(np.std(diff_gray))

        local_discrepancy = 0.0
        face_mean = global_mean
        bg_mean = global_mean

        if face_bbox is not None and len(face_bbox) == 4:
            x, y, w, h = face_bbox
            img_h, img_w = diff_gray.shape
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

            if (x2 > x1) and (y2 > y1):
                face_crop = diff_gray[y1:y2, x1:x2]
                face_mean = float(np.mean(face_crop))

                # Background mask
                mask_bg = np.ones_like(diff_gray, dtype=bool)
                mask_bg[y1:y2, x1:x2] = False
                if np.any(mask_bg):
                    bg_mean = float(np.mean(diff_gray[mask_bg]))
                    # Discrepancy between face compression error and background
                    local_discrepancy = abs(face_mean - bg_mean) / max(1.0, bg_mean)

        # Baseline anomaly score
        anomaly_score = 0.10
        # High local discrepancy between face and background signals splicing
        if local_discrepancy > 0.35:
            anomaly_score += min(0.50, (local_discrepancy - 0.35) * 1.0 + 0.25)
        # Very high global variance in compression errors indicates multi-source splicing
        if global_std > 8.0:
            anomaly_score += min(0.30, (global_std - 8.0) * 0.05)

        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))
        anomaly_detected = bool(anomaly_score > 0.45)

        description = (
            f"ELA compression: Global error mean {global_mean:.1f} (std {global_std:.1f}), "
            f"face/background discrepancy {local_discrepancy:.2f}. "
        )
        if anomaly_detected:
            description += "Significant localized compression inconsistency flagged."
        else:
            description += "Uniform compression levels observed across regions."

        return {
            "ela_score": round(anomaly_score, 4),
            "global_error_mean": round(global_mean, 2),
            "global_error_std": round(global_std, 2),
            "local_discrepancy": round(local_discrepancy, 4),
            "anomaly_detected": anomaly_detected,
            "description": description,
        }


class LaplacianEdgeAnalyzer:
    """
    Evaluates edge boundary transitions and micro-texture variance across
    facial regions and global image planes to detect boundary blending artifacts.
    """

    def analyze(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Args:
            image: BGR numpy image array.
            face_bbox: Optional (x, y, w, h) bounding box of detected face.
        Returns:
            Dict containing edge consistency metrics and description.
        """
        if image is None or image.size == 0:
            return {
                "edge_score": 0.0,
                "laplacian_var": 0.0,
                "anomaly_detected": False,
                "description": "Empty image provided to edge analyzer.",
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())

        boundary_discontinuity = 0.0
        if face_bbox is not None and len(face_bbox) == 4:
            x, y, w, h = face_bbox
            img_h, img_w = gray.shape
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)

            if (x2 > x1 + 10) and (y2 > y1 + 10):
                # Sample border perimeter (band of width 6 pixels around bbox)
                pad = 6
                bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
                bx2, by2 = min(img_w, x2 + pad), min(img_h, y2 + pad)

                perimeter_mask = np.zeros_like(gray, dtype=bool)
                perimeter_mask[by1:by2, bx1:bx2] = True
                perimeter_mask[y1:y2, x1:x2] = False

                inner_face_lap = lap[y1:y2, x1:x2]
                inner_var = float(inner_face_lap.var()) if inner_face_lap.size > 0 else lap_var

                if np.any(perimeter_mask):
                    border_lap = lap[perimeter_mask]
                    border_var = float(border_lap.var())
                    # Discontinuity ratio between face border gradient and interior face gradient
                    boundary_discontinuity = abs(inner_var - border_var) / max(10.0, inner_var)

        anomaly_score = 0.10
        if boundary_discontinuity > 0.40:
            anomaly_score += min(0.50, (boundary_discontinuity - 0.40) * 1.2 + 0.25)
        if lap_var < 35.0:  # Noticeable smoothing / airbrushing typical of synthetic generation
            anomaly_score += min(0.30, (35.0 - lap_var) * 0.02)

        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))
        anomaly_detected = bool(anomaly_score > 0.45)

        description = f"Edge & Texture: Global Laplacian variance {lap_var:.1f}, border discontinuity {boundary_discontinuity:.2f}. "
        if anomaly_detected:
            description += "Boundary transition mismatch detected around facial contour."
        else:
            description += "Smooth, continuous edge transitions consistent with natural optics."

        return {
            "edge_score": round(anomaly_score, 4),
            "laplacian_var": round(lap_var, 2),
            "boundary_discontinuity": round(boundary_discontinuity, 4),
            "anomaly_detected": anomaly_detected,
            "description": description,
        }


class ForensicEvidenceSuite:
    """
    Executes independent genuine forensic analyzers on the target media.
    Provides verifiable, non-fabricated forensic evidence items.
    """

    def __init__(self):
        self.fft_analyzer = FFTSpectralAnalyzer()
        self.ela_analyzer = ErrorLevelAnalyzer()
        self.edge_analyzer = LaplacianEdgeAnalyzer()

    def evaluate(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Runs all independent forensic analyses and computes a composite anomaly score.
        """
        fft_res = self.fft_analyzer.analyze(image)
        ela_res = self.ela_analyzer.analyze(image, face_bbox=face_bbox)
        edge_res = self.edge_analyzer.analyze(image, face_bbox=face_bbox)

        # Genuine evidence items based strictly on measurements
        evidence_items = []
        if fft_res.get("anomaly_detected"):
            evidence_items.append(f"FFT Spectrum: {fft_res.get('description')}")
        if ela_res.get("anomaly_detected"):
            evidence_items.append(f"Compression ELA: {ela_res.get('description')}")
        if edge_res.get("anomaly_detected"):
            evidence_items.append(f"Boundary Continuity: {edge_res.get('description')}")

        scores = [
            fft_res.get("fft_score", 0.0),
            ela_res.get("ela_score", 0.0),
            edge_res.get("edge_score", 0.0),
        ]
        # Weighted aggregate: max component contributes strongly to prevent false negatives
        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        composite_score = float(0.6 * max_score + 0.4 * mean_score)

        return {
            "composite_forensic_score": round(composite_score, 4),
            "fft_analysis": fft_res,
            "ela_analysis": ela_res,
            "edge_analysis": edge_res,
            "evidence_items": evidence_items,
            "suspicious_signal_count": len(evidence_items),
        }
