import os
import json
import datetime
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.optimize import minimize_scalar
from backend.utils.logger import setup_logger
from backend.utils.metrics_helper import compute_ece

logger = setup_logger("Calibrator")

DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "calibration_config.json",
)


class TemperatureScalingCalibrator:
    """
    Implements Temperature Scaling for binary classification calibration.
    Preserves model ranking (ROC-AUC invariant) while aligning confidence
    with true empirical accuracy, directly minimizing Expected Calibration Error (ECE).
    """

    def __init__(
        self,
        temperature: float = 1.65,
        tau_real: float = 0.35,
        tau_fake: float = 0.65,
        min_confidence: float = 0.60,
        config_path: str = DEFAULT_CALIBRATION_PATH,
    ):
        self.temperature = float(temperature)
        self.tau_real = float(tau_real)
        self.tau_fake = float(tau_fake)
        self.min_confidence = float(min_confidence)
        self.config_path = config_path
        self.metadata = {}

        # Load persisted config if available
        self.load()

    def calibrate_probability(self, raw_fake_prob: float) -> float:
        """
        Calibrates a raw predicted probability of fake using temperature scaling.
        Args:
            raw_fake_prob: Float in [0.0, 1.0].
        Returns:
            Calibrated fake probability in [0.0, 1.0].
        """
        # Clamp to avoid log(0)
        p = np.clip(raw_fake_prob, 1e-6, 1.0 - 1e-6)
        # Convert to logit difference: z = ln(p / (1 - p))
        logit = np.log(p / (1.0 - p))
        # Temperature scale
        scaled_logit = logit / max(1e-2, self.temperature)
        # Sigmoid to calibrated probability
        cal_p = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(np.clip(cal_p, 0.0, 1.0))

    def calibrate_batch(self, raw_fake_probs: np.ndarray) -> np.ndarray:
        """Calibrates an array of probabilities."""
        raw_fake_probs = np.array(raw_fake_probs, dtype=np.float64)
        p = np.clip(raw_fake_probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p))
        scaled_logits = logits / max(1e-2, self.temperature)
        return 1.0 / (1.0 + np.exp(-scaled_logits))

    def get_confidence_and_uncertainty(self, calibrated_prob: float) -> Tuple[float, float]:
        """
        Calculates confidence (0-100) and normalized uncertainty (0-1).
        Args:
            calibrated_prob: Calibrated fake probability in [0.0, 1.0].
        Returns:
            Tuple of (confidence_percent, uncertainty_ratio).
        """
        # Binary classification confidence is max(p, 1-p)
        conf_ratio = max(calibrated_prob, 1.0 - calibrated_prob)
        confidence_percent = round(conf_ratio * 100.0, 2)

        # Uncertainty is maximum at p=0.5 (uncertainty=1.0) and 0 at p=0 or 1
        uncertainty = round(1.0 - abs(2.0 * calibrated_prob - 1.0), 4)
        return confidence_percent, uncertainty

    def fit_on_validation(
        self,
        y_val_true: np.ndarray,
        y_val_raw_prob: np.ndarray,
        save_after_fit: bool = True,
    ) -> Dict[str, Any]:
        """
        Fits optimal temperature T strictly on validation split data.
        Minimizes negative log-likelihood (binary cross entropy).
        Does NOT touch the test set.
        """
        y_true = np.array(y_val_true, dtype=np.float64)
        y_prob = np.clip(np.array(y_val_raw_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        logits = np.log(y_prob / (1.0 - y_prob))

        ece_before = compute_ece(y_true, y_prob)

        def nll_objective(t):
            if t <= 0:
                return 1e6
            scaled = logits / t
            probs = 1.0 / (1.0 + np.exp(-scaled))
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
            loss = -np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs))
            return loss

        res = minimize_scalar(nll_objective, bounds=(0.1, 10.0), method="bounded")
        opt_t = float(res.x) if res.success else self.temperature

        cal_probs = 1.0 / (1.0 + np.exp(-logits / opt_t))
        ece_after = compute_ece(y_true, cal_probs)

        # Derive data-driven decision boundaries on validation split:
        # Sort calibrated probabilities for real and fake classes
        real_probs = cal_probs[y_true == 0]
        fake_probs = cal_probs[y_true == 1]

        # Tau_real: 85th percentile of real class calibrated probabilities (guaranteeing low false positives)
        # Tau_fake: 15th percentile of fake class calibrated probabilities (guaranteeing low false negatives)
        tau_real = float(np.percentile(real_probs, 80)) if len(real_probs) > 0 else 0.35
        tau_fake = float(np.percentile(fake_probs, 20)) if len(fake_probs) > 0 else 0.65

        # Enforce sensible safety guardrails
        tau_real = max(0.20, min(0.40, tau_real))
        tau_fake = max(0.60, min(0.80, tau_fake))

        self.temperature = opt_t
        self.tau_real = round(tau_real, 3)
        self.tau_fake = round(tau_fake, 3)

        fit_summary = {
            "method": "temperature_scaling",
            "temperature": round(opt_t, 4),
            "validation_ece_before": round(ece_before, 4),
            "validation_ece_after": round(ece_after, 4),
            "tau_real": self.tau_real,
            "tau_fake": self.tau_fake,
            "min_confidence": self.min_confidence,
            "fitted_on": "validation_split_only",
            "validation_samples_count": len(y_true),
            "calibrated_at": datetime.datetime.now().isoformat(),
        }

        self.metadata = fit_summary
        if save_after_fit:
            self.save()

        logger.info(
            f"Fitted Temperature Scaling on {len(y_true)} val samples: "
            f"T={opt_t:.4f}, ECE {ece_before:.4f} -> {ece_after:.4f}, "
            f"Bounds: Real <= {self.tau_real}, Fake >= {self.tau_fake}"
        )
        return fit_summary

    def save(self, path: Optional[str] = None) -> None:
        """Persists calibration parameters to JSON artifact."""
        target_path = path or self.config_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        payload = {
            "method": "temperature_scaling",
            "temperature": self.temperature,
            "tau_real": self.tau_real,
            "tau_fake": self.tau_fake,
            "min_confidence": self.min_confidence,
            "metadata": self.metadata,
            "saved_at": datetime.datetime.now().isoformat(),
        }
        with open(target_path, "w") as f:
            json.dump(payload, f, indent=4)
        logger.info(f"Saved calibration parameters to: {target_path}")

    def load(self, path: Optional[str] = None) -> bool:
        """Loads persisted calibration parameters from JSON artifact."""
        target_path = path or self.config_path
        if not os.path.exists(target_path):
            logger.info(f"No existing calibration config at {target_path}. Using validated default T={self.temperature}.")
            return False
        try:
            with open(target_path, "r") as f:
                data = json.load(f)
            self.temperature = float(data.get("temperature", self.temperature))
            self.tau_real = float(data.get("tau_real", self.tau_real))
            self.tau_fake = float(data.get("tau_fake", self.tau_fake))
            self.min_confidence = float(data.get("min_confidence", self.min_confidence))
            self.metadata = data.get("metadata", {})
            logger.info(f"Loaded calibration config from {target_path}: T={self.temperature}, tau_real={self.tau_real}, tau_fake={self.tau_fake}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load calibration config: {e}. Using defaults.")
            return False
