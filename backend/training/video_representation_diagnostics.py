import os
import csv
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, List, Dict
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torchvision.models as models
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.models.video_model import VideoModel
from backend.preprocessing.face_detector import FaceDetector

logger = setup_logger("VideoRepresentationDiagnostics")

class DiagnosticsPipeline:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.manifest_path = os.path.join(workspace_dir, "backend", "training", "dataset_manifest.csv")
        self.pilot_manifest_path = os.path.join(workspace_dir, "backend", "training", "results", "video", "pilot_subset_manifest.csv")
        self.output_dir = os.path.join(workspace_dir, "backend", "training", "results", "video", "diagnostics")
        self.plots_dir = os.path.join(self.output_dir, "plots")
        self.frames_dir = os.path.join(self.output_dir, "representative_frames")
        
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        self.detector = FaceDetector()
        self.target_size = PipelineConfig.IMAGE_SIZE
        
        # Paths to checkpoints
        self.scratch_checkpoint = os.path.join(workspace_dir, "backend", "models", "weights", "runs", "video_standard_20260816_210343.pth")
        self.pretrained_checkpoint = os.path.join(workspace_dir, "backend", "models", "weights", "runs", "video_standard_20260816_211329.pth")

        # In-memory video cache
        self.video_cache = {}

    def load_pilot_samples(self) -> Tuple[List[Dict], List[Dict]]:
        """Loads pilot train and validation samples, strictly ignoring test samples."""
        train_samples = []
        val_samples = []
        
        if not os.path.exists(self.pilot_manifest_path):
            raise FileNotFoundError(f"Pilot subset manifest not found at: {self.pilot_manifest_path}")
            
        with open(self.pilot_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row["split"].lower()
                sample = {
                    "file_path": row["file_path"],
                    "label": int(row["label"]),
                    "split": split,
                    "dataset_source": row.get("dataset_source", "")
                }
                if split == "train":
                    train_samples.append(sample)
                elif split == "val":
                    val_samples.append(sample)
                    
        logger.info(f"Loaded {len(train_samples)} Train samples, {len(val_samples)} Validation samples from pilot manifest.")
        return train_samples, val_samples

    def preprocess_and_cache_video(self, file_path: str) -> Dict:
        """
        Extracts 10 frames from the video, runs face detection, crops, masks backgrounds,
        and saves processed uint8 frames and metadata in a cache.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")
            
        cap = None
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {file_path}")
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = 0.0
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                duration = total_frames / fps
                
            file_size = os.path.getsize(file_path)
            bitrate = (file_size * 8) / max(0.1, duration)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Max frames to extract for caching is 10
            max_frames = 10
            if total_frames > 0:
                target_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
            else:
                target_indices = [i * 15 for i in range(max_frames)]
                
            face_frames = []
            bg_frames = []
            detections = []
            face_detected_count = 0
            crop_widths = []
            crop_heights = []
            crop_aspect_ratios = []
            brightness_list = []
            rgb_means = []
            rgb_stds = []
            
            for idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                        
                faces = self.detector.detect_faces(frame)
                face_detected = len(faces) > 0
                
                if face_detected:
                    face_detected_count += 1
                    bbox = faces[0]
                    detections.append(bbox)
                    x, y, w, h = bbox
                    crop_widths.append(w)
                    crop_heights.append(h)
                    crop_aspect_ratios.append(w / max(1, h))
                    
                    # Crop face
                    face_crop = self.detector.crop_face(frame, bbox)
                    
                    # Mask out face for background shortcut
                    bg_frame = frame.copy()
                    bg_frame[y:y+h, x:x+w] = 0
                else:
                    face_crop = frame
                    bg_frame = np.zeros_like(frame)
                    
                processed_face = cv2.resize(face_crop, self.target_size)
                face_rgb = cv2.cvtColor(processed_face, cv2.COLOR_BGR2RGB)
                
                processed_bg = cv2.resize(bg_frame, self.target_size)
                bg_rgb = cv2.cvtColor(processed_bg, cv2.COLOR_BGR2RGB)
                
                # Artifact stats
                brightness_list.append(float(np.mean(face_rgb)))
                rgb_means.append(np.mean(face_rgb, axis=(0, 1)).tolist())
                rgb_stds.append(np.std(face_rgb, axis=(0, 1)).tolist())
                
                face_frames.append(face_rgb)
                bg_frames.append(bg_rgb)
            
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        
        # Fallback if zero frames read
        if not face_frames:
            dummy = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
            face_frames = [dummy] * max_frames
            bg_frames = [dummy] * max_frames
            brightness_list = [0.0]
            rgb_means = [[0.0, 0.0, 0.0]]
            rgb_stds = [[0.0, 0.0, 0.0]]
            
        while len(face_frames) < max_frames:
            face_frames.append(face_frames[-1].copy())
            bg_frames.append(bg_frames[-1].copy())
            
        success_rate = face_detected_count / max(1, len(target_indices))
        fallback_rate = 1.0 - success_rate
        
        return {
            "face_frames": face_frames, # list of 10 uint8 RGB arrays
            "bg_frames": bg_frames,     # list of 10 uint8 RGB arrays
            "success_rate": success_rate,
            "fallback_rate": fallback_rate,
            "avg_crop_width": float(np.mean(crop_widths)) if crop_widths else 0.0,
            "avg_crop_height": float(np.mean(crop_heights)) if crop_heights else 0.0,
            "avg_aspect_ratio": float(np.mean(crop_aspect_ratios)) if crop_aspect_ratios else 0.0,
            "bbox_stats": detections[0] if detections else [0, 0, 0, 0],
            "brightness": float(np.mean(brightness_list)),
            "rgb_mean": np.mean(rgb_means, axis=0).tolist(),
            "rgb_std": np.mean(rgb_stds, axis=0).tolist(),
            "frame_width": width,
            "frame_height": height,
            "duration": duration,
            "file_size": file_size,
            "bitrate": bitrate,
            "last_processed_frame": cv2.cvtColor(face_frames[-1], cv2.COLOR_RGB2BGR)
        }

    def get_features_from_cache(self, file_path: str, model: nn.Module, max_frames: int, 
                                 use_background: bool = False) -> Dict:
        """Retrieves preprocessed frames from cache, formats them, and runs model backbone feature extraction."""
        cache_data = self.video_cache[file_path]
        frames_list = cache_data["bg_frames"] if use_background else cache_data["face_frames"]
        
        # Subsample frames to match max_frames
        indices = np.linspace(0, len(frames_list) - 1, max_frames, dtype=int).tolist()
        
        tensors = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        for idx in indices:
            frame_rgb = frames_list[idx]
            norm_frame = (frame_rgb.astype(np.float32) / 255.0 - mean) / std
            trans_frame = np.transpose(norm_frame, (2, 0, 1))
            tensors.append(torch.tensor(trans_frame, dtype=torch.float32))
            
        video_tensor = torch.stack(tensors, dim=0).unsqueeze(0) # shape (1, max_frames, 3, H, W)
        
        model.eval()
        with torch.no_grad():
            b, s, c, h_t, w_t = video_tensor.size()
            flat_x = video_tensor.view(b * s, c, h_t, w_t)
            features_flat = model.backbone(flat_x)
            seq_features = features_flat.view(s, -1).cpu().numpy() # (max_frames, 512)
            
            # LSTM hidden state (only for face features of normal sequence models)
            lstm_rep = None
            if not use_background and hasattr(model, "lstm"):
                features = features_flat.view(b, s, -1)
                lstm_out, _ = model.lstm(features)
                lstm_rep = lstm_out[0, -1, :].cpu().numpy() # (256,)
                
        return {
            "features": seq_features,
            "lstm_rep": lstm_rep
        }

    def run_pca_tsne(self, features: np.ndarray, labels: np.ndarray, name: str):
        """Generates PCA and t-SNE plots for visualization."""
        if not HAS_MATPLOTLIB:
            logger.warning(f"Matplotlib not available. Skipping plot generation for {name}.")
            return
        if features.shape[0] < 2:
            return
            
        # PCA
        pca = PCA(n_components=2, random_state=42)
        pca_feats = pca.fit_transform(features)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(pca_feats[labels == 0, 0], pca_feats[labels == 0, 1], label="REAL", alpha=0.7, color="blue")
        plt.scatter(pca_feats[labels == 1, 0], pca_feats[labels == 1, 1], label="FAKE", alpha=0.7, color="red")
        plt.title(f"PCA - {name}")
        plt.legend()
        
        # t-SNE
        perplexity = min(30, max(2, features.shape[0] // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        tsne_feats = tsne.fit_transform(features)
        
        plt.subplot(1, 2, 2)
        plt.scatter(tsne_feats[labels == 0, 0], tsne_feats[labels == 0, 1], label="REAL", alpha=0.7, color="blue")
        plt.scatter(tsne_feats[labels == 1, 0], tsne_feats[labels == 1, 1], label="FAKE", alpha=0.7, color="red")
        plt.title(f"t-SNE - {name}")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"{name.lower().replace(' ', '_')}_scatter.png"))
        plt.close()

    def evaluate_separability(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculates distance-based class separability metrics."""
        real_feats = features[labels == 0]
        fake_feats = features[labels == 1]
        
        if len(real_feats) == 0 or len(fake_feats) == 0:
            return {
                "euclidean_centroid_distance": 0.0,
                "cosine_centroid_distance": 0.0,
                "within_class_variance": 0.0,
                "between_class_variance": 0.0
            }
            
        real_centroid = np.mean(real_feats, axis=0)
        fake_centroid = np.mean(fake_feats, axis=0)
        
        # Euclidean centroid distance
        euc_dist = float(np.linalg.norm(real_centroid - fake_centroid))
        
        # Cosine centroid distance
        norm_r = np.linalg.norm(real_centroid)
        norm_f = np.linalg.norm(fake_centroid)
        if norm_r > 0 and norm_f > 0:
            cosine_dist = float(1.0 - np.dot(real_centroid, fake_centroid) / (norm_r * norm_f))
        else:
            cosine_dist = 1.0
            
        # Variance
        real_var = np.mean(np.var(real_feats, axis=0))
        fake_var = np.mean(np.var(fake_feats, axis=0))
        within_class_var = float((real_var * len(real_feats) + fake_var * len(fake_feats)) / len(features))
        
        overall_centroid = np.mean(features, axis=0)
        between_class_var = float(
            (len(real_feats) * np.sum((real_centroid - overall_centroid)**2) + 
             len(fake_feats) * np.sum((fake_centroid - overall_centroid)**2)) / len(features)
        )
        
        return {
            "euclidean_centroid_distance": euc_dist,
            "cosine_centroid_distance": cosine_dist,
            "within_class_variance": within_class_var,
            "between_class_variance": between_class_var
        }

    def train_and_eval_probes(self, train_x: np.ndarray, train_y: np.ndarray, 
                              val_x: np.ndarray, val_y: np.ndarray) -> Dict:
        """Trains linear probes on train embeddings and evaluates on validation embeddings."""
        probes = {
            "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "Linear SVM": LinearSVC(max_iter=2000, dual=False, C=1.0, random_state=42),
            "Ridge": RidgeClassifier(random_state=42),
            "Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
        }
        
        results = {}
        for name, clf in probes.items():
            try:
                clf.fit(train_x, train_y)
                preds = clf.predict(val_x)
                
                # Check for probabilities
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(val_x)[:, 1]
                elif hasattr(clf, "decision_function"):
                    dec = clf.decision_function(val_x)
                    probs = 1.0 / (1.0 + np.exp(-dec)) # sigmoid fallback
                else:
                    probs = preds.astype(float)
                    
                acc = float(accuracy_score(val_y, preds))
                macro_f1 = float(f1_score(val_y, preds, average="macro", zero_division=0))
                real_rec = float(recall_score(val_y, preds, pos_label=0, zero_division=0))
                fake_rec = float(recall_score(val_y, preds, pos_label=1, zero_division=0))
                
                try:
                    auc = float(roc_auc_score(val_y, probs))
                except ValueError:
                    auc = 0.5
                    
                cm = confusion_matrix(val_y, preds).tolist()
                
                results[name] = {
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "real_recall": real_rec,
                    "fake_recall": fake_rec,
                    "roc_auc": auc,
                    "confusion_matrix": cm
                }
            except Exception as e:
                logger.error(f"Error training/evaluating probe {name}: {str(e)}")
                results[name] = {
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "real_recall": 0.0,
                    "fake_recall": 0.0,
                    "roc_auc": 0.5,
                    "confusion_matrix": [[0, 0], [0, 0]]
                }
        return results

    def run_diagnostics(self):
        """Runs the entire diagnostic process."""
        train_samples, val_samples = self.load_pilot_samples()
        
        # Load learned Standard VideoModel
        logger.info("Initializing VideoModel spatial & temporal backbones...")
        scratch_model = VideoModel(num_classes=2, pretrained=False)
        if os.path.exists(self.scratch_checkpoint):
            scratch_model.load_weights(self.scratch_checkpoint)
            logger.info("Successfully loaded learned scratch VideoModel weights.")
        else:
            logger.warning(f"Learned scratch VideoModel checkpoint not found at: {self.scratch_checkpoint}. Using randomly initialized backbone.")
            
        pretrained_model = VideoModel(num_classes=2, pretrained=True)
        if os.path.exists(self.pretrained_checkpoint):
            pretrained_model.load_weights(self.pretrained_checkpoint)
            logger.info("Successfully loaded learned pretrained VideoModel weights.")
        else:
            logger.warning(f"Learned pretrained VideoModel checkpoint not found at: {self.pretrained_checkpoint}. Using standard pretrained model.")

        # Pretrained ResNet-18 directly from PyTorch hub (frozen reference)
        raw_resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        raw_resnet18.fc = nn.Identity()
        raw_resnet18.eval()
        
        # Check ResNet-50 availability locally
        resnet50_available = False
        try:
            raw_resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            raw_resnet50.fc = nn.Identity()
            raw_resnet50.eval()
            resnet50_available = True
            logger.info("Pretrained ResNet-50 is available locally.")
        except Exception:
            logger.info("Pretrained ResNet-50 is unavailable locally. Skipping.")
            
        # STEP 1: PREPROCESS AND CACHE ALL VIDEOS (10 frames each)
        logger.info("Preprocessing and caching all videos in pilot subset...")
        
        face_audit_stats = []
        artifact_audit_stats = []
        real_saved = 0
        fake_saved = 0
        
        total_samples = len(train_samples) + len(val_samples)
        for i, sample in enumerate(train_samples + val_samples):
            file_path = sample["file_path"]
            label = sample["label"]
            split = sample["split"]
            
            if i % 20 == 0:
                logger.info(f"Caching progress: {i}/{total_samples} videos processed...")
                
            try:
                res = self.preprocess_and_cache_video(file_path)
                self.video_cache[file_path] = res
                
                # Perform audits on validation subset
                if split == "val":
                    face_audit_stats.append({
                        "label": label,
                        "success_rate": res["success_rate"],
                        "fallback_rate": res["fallback_rate"],
                        "avg_crop_width": res["avg_crop_width"],
                        "avg_crop_height": res["avg_crop_height"],
                        "avg_aspect_ratio": res["avg_aspect_ratio"],
                        "bbox": res["bbox_stats"]
                    })
                    
                    artifact_audit_stats.append({
                        "label": label,
                        "brightness": res["brightness"],
                        "rgb_mean": res["rgb_mean"],
                        "rgb_std": res["rgb_std"],
                        "width": res["frame_width"],
                        "height": res["frame_height"],
                        "duration": res["duration"],
                        "file_size": res["file_size"],
                        "bitrate": res["bitrate"]
                    })
                    
                    # Save a few representative frames
                    if label == 0 and real_saved < 5:
                        cv2.imwrite(os.path.join(self.frames_dir, f"real_val_video_{real_saved}.jpg"), res["last_processed_frame"])
                        real_saved += 1
                    elif label == 1 and fake_saved < 5:
                        cv2.imwrite(os.path.join(self.frames_dir, f"fake_val_video_{fake_saved}.jpg"), res["last_processed_frame"])
                        fake_saved += 1
                        
            except Exception as e:
                logger.warning(f"Failed to process sample {file_path}: {str(e)}")
                
        logger.info("Finished preprocessing cache. Running diagnostic runs...")
        
        # 1. FACE PREPROCESSING AUDIT REPORT
        real_success = [s["success_rate"] for s in face_audit_stats if s["label"] == 0]
        fake_success = [s["success_rate"] for s in face_audit_stats if s["label"] == 1]
        real_fallback = [s["fallback_rate"] for s in face_audit_stats if s["label"] == 0]
        fake_fallback = [s["fallback_rate"] for s in face_audit_stats if s["label"] == 1]
        
        real_widths = [s["avg_crop_width"] for s in face_audit_stats if s["label"] == 0 and s["success_rate"] > 0]
        fake_widths = [s["avg_crop_width"] for s in face_audit_stats if s["label"] == 1 and s["success_rate"] > 0]
        real_heights = [s["avg_crop_height"] for s in face_audit_stats if s["label"] == 0 and s["success_rate"] > 0]
        fake_heights = [s["avg_crop_height"] for s in face_audit_stats if s["label"] == 1 and s["success_rate"] > 0]
        
        real_ratios = [s["avg_aspect_ratio"] for s in face_audit_stats if s["label"] == 0 and s["success_rate"] > 0]
        fake_ratios = [s["avg_aspect_ratio"] for s in face_audit_stats if s["label"] == 1 and s["success_rate"] > 0]
        
        face_detection_report = {
            "REAL": {
                "success_rate": float(np.mean(real_success)) if real_success else 0.0,
                "fallback_rate": float(np.mean(real_fallback)) if real_fallback else 1.0,
                "avg_crop_width": float(np.mean(real_widths)) if real_widths else 0.0,
                "avg_crop_height": float(np.mean(real_heights)) if real_heights else 0.0,
                "avg_aspect_ratio": float(np.mean(real_ratios)) if real_ratios else 0.0
            },
            "FAKE": {
                "success_rate": float(np.mean(fake_success)) if fake_success else 0.0,
                "fallback_rate": float(np.mean(fake_fallback)) if fake_fallback else 1.0,
                "avg_crop_width": float(np.mean(fake_widths)) if fake_widths else 0.0,
                "avg_crop_height": float(np.mean(fake_heights)) if fake_heights else 0.0,
                "avg_aspect_ratio": float(np.mean(fake_ratios)) if fake_ratios else 0.0
            }
        }
        
        with open(os.path.join(self.output_dir, "face_detection_report.json"), "w") as f:
            json.dump(face_detection_report, f, indent=4)
            
        # 2. DATASET ARTIFACT AUDIT REPORT
        real_bright = [s["brightness"] for s in artifact_audit_stats if s["label"] == 0]
        fake_bright = [s["brightness"] for s in artifact_audit_stats if s["label"] == 1]
        
        real_durations = [s["duration"] for s in artifact_audit_stats if s["label"] == 0]
        fake_durations = [s["duration"] for s in artifact_audit_stats if s["label"] == 1]
        
        real_sizes = [s["file_size"] for s in artifact_audit_stats if s["label"] == 0]
        fake_sizes = [s["file_size"] for s in artifact_audit_stats if s["label"] == 1]
        
        real_bitrates = [s["bitrate"] for s in artifact_audit_stats if s["label"] == 0]
        fake_bitrates = [s["bitrate"] for s in artifact_audit_stats if s["label"] == 1]
        
        real_widths = [s["width"] for s in artifact_audit_stats if s["label"] == 0]
        fake_widths = [s["width"] for s in artifact_audit_stats if s["label"] == 1]
        
        real_heights = [s["height"] for s in artifact_audit_stats if s["label"] == 0]
        fake_heights = [s["height"] for s in artifact_audit_stats if s["label"] == 1]
        
        frame_sampling_report = {
            "brightness": {
                "real_mean": float(np.mean(real_bright)) if real_bright else 0.0,
                "fake_mean": float(np.mean(fake_bright)) if fake_bright else 0.0
            },
            "duration": {
                "real_mean": float(np.mean(real_durations)) if real_durations else 0.0,
                "fake_mean": float(np.mean(fake_durations)) if fake_durations else 0.0
            },
            "file_size": {
                "real_mean": float(np.mean(real_sizes)) if real_sizes else 0.0,
                "fake_mean": float(np.mean(fake_sizes)) if fake_sizes else 0.0
            },
            "bitrate": {
                "real_mean": float(np.mean(real_bitrates)) if real_bitrates else 0.0,
                "fake_mean": float(np.mean(fake_bitrates)) if fake_bitrates else 0.0
            },
            "resolution": {
                "real_width_mean": float(np.mean(real_widths)) if real_widths else 0.0,
                "real_height_mean": float(np.mean(real_heights)) if real_heights else 0.0,
                "fake_width_mean": float(np.mean(fake_widths)) if fake_widths else 0.0,
                "fake_height_mean": float(np.mean(fake_heights)) if fake_heights else 0.0
            }
        }
        
        with open(os.path.join(self.output_dir, "frame_sampling_report.json"), "w") as f:
            json.dump(frame_sampling_report, f, indent=4)
            
        # Helper to extract features from cached samples
        def extract_dataset_features(model, max_frames, use_background=False):
            tr_feats, tr_lstm, tr_labels = [], [], []
            v_feats, v_lstm, v_labels = [], [], []
            
            for s in train_samples:
                if s["file_path"] in self.video_cache:
                    res = self.get_features_from_cache(s["file_path"], model, max_frames, use_background)
                    tr_feats.append(res["features"])
                    if res["lstm_rep"] is not None:
                        tr_lstm.append(res["lstm_rep"])
                    tr_labels.append(s["label"])
                    
            for s in val_samples:
                if s["file_path"] in self.video_cache:
                    res = self.get_features_from_cache(s["file_path"], model, max_frames, use_background)
                    v_feats.append(res["features"])
                    if res["lstm_rep"] is not None:
                        v_lstm.append(res["lstm_rep"])
                    v_labels.append(s["label"])
                    
            return (
                np.array(tr_feats), np.array(tr_lstm) if tr_lstm else None, np.array(tr_labels),
                np.array(v_feats), np.array(v_lstm) if v_lstm else None, np.array(v_labels)
            )

        # 3. REPRESENTATION ABLATION & LINEAR PROBES ON LEARNED PRETRAINED EMBEDDINGS (5 frames)
        tr_face, tr_lstm, train_labels, v_face, v_lstm, val_labels = extract_dataset_features(pretrained_model, max_frames=5)
        
        mid_idx = 5 // 2
        train_mid = tr_face[:, mid_idx, :]
        val_mid = v_face[:, mid_idx, :]
        
        train_mean = np.mean(tr_face, axis=1)
        val_mean = np.mean(v_face, axis=1)
        
        train_max = np.max(tr_face, axis=1)
        val_max = np.max(v_face, axis=1)
        
        ablation_configs = {
            "Middle Frame Spatial (512-dim)": (train_mid, val_mid),
            "Mean Pooled Spatial (512-dim)": (train_mean, val_mean),
            "Max Pooled Spatial (512-dim)": (train_max, val_max)
        }
        if tr_lstm is not None and v_lstm is not None:
            ablation_configs["LSTM Hidden Rep (256-dim)"] = (tr_lstm, v_lstm)
            
        ablation_results = {}
        embedding_statistics = {}
        
        for name, (tr_x, v_x) in ablation_configs.items():
            sep = self.evaluate_separability(v_x, val_labels)
            embedding_statistics[name] = sep
            self.run_pca_tsne(v_x, val_labels, name)
            probes_res = self.train_and_eval_probes(tr_x, train_labels, v_x, val_labels)
            ablation_results[name] = probes_res
            
        # 4. BACKGROUND SHORTCUT TEST
        tr_bg, _, _, v_bg, _, _ = extract_dataset_features(pretrained_model, max_frames=5, use_background=True)
        train_bg_mean = np.mean(tr_bg, axis=1)
        val_bg_mean = np.mean(v_bg, axis=1)
        bg_probes_res = self.train_and_eval_probes(train_bg_mean, train_labels, val_bg_mean, val_labels)
        ablation_results["Masked Background Mean Pooled (512-dim)"] = bg_probes_res
        
        # Save ablation results
        with open(os.path.join(self.output_dir, "ablation_results.json"), "w") as f:
            json.dump(ablation_results, f, indent=4)
            
        with open(os.path.join(self.output_dir, "embedding_statistics.json"), "w") as f:
            json.dump(embedding_statistics, f, indent=4)
            
        # 5. FRAME-COUNT ABLATION: 3, 5, 10 frames
        frame_counts = [3, 5, 10]
        frame_count_results = {}
        
        for fc in frame_counts:
            logger.info(f"Extracting features for Frame Count Ablation: {fc} frames...")
            tr_fc, _, _, v_fc, _, _ = extract_dataset_features(pretrained_model, max_frames=fc)
            tr_fc_mean = np.mean(tr_fc, axis=1)
            v_fc_mean = np.mean(v_fc, axis=1)
            fc_probes = self.train_and_eval_probes(tr_fc_mean, train_labels, v_fc_mean, val_labels)
            frame_count_results[f"{fc}_frames"] = fc_probes
            
        # 6. BACKBONE ABLATION
        # A. Scratch model
        logger.info("Extracting features for Backbone Ablation: Learned Scratch ResNet-18...")
        tr_sc, _, _, v_sc, _, _ = extract_dataset_features(scratch_model, max_frames=5)
        tr_sc_mean = np.mean(tr_sc, axis=1)
        v_sc_mean = np.mean(v_sc, axis=1)
        scratch_probes = self.train_and_eval_probes(tr_sc_mean, train_labels, v_sc_mean, val_labels)
        
        # B. Raw PyTorch Pretrained ResNet-18
        logger.info("Extracting features for Backbone Ablation: Raw Pretrained ResNet-18...")
        class RawWrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            def forward(self, x):
                return self.backbone(x)
                
        raw_wrapper = RawWrapper(raw_resnet18)
        tr_raw, _, _, v_raw, _, _ = extract_dataset_features(raw_wrapper, max_frames=5)
        tr_raw_mean = np.mean(tr_raw, axis=1)
        v_raw_mean = np.mean(v_raw, axis=1)
        raw_probes = self.train_and_eval_probes(tr_raw_mean, train_labels, v_raw_mean, val_labels)
        
        # C. Raw PyTorch Pretrained ResNet-50
        resnet50_probes = {}
        if resnet50_available:
            logger.info("Extracting features for Backbone Ablation: Raw Pretrained ResNet-50...")
            raw_wrapper_50 = RawWrapper(raw_resnet50)
            tr_50, _, _, v_50, _, _ = extract_dataset_features(raw_wrapper_50, max_frames=5)
            tr_50_mean = np.mean(tr_50, axis=1)
            v_50_mean = np.mean(v_50, axis=1)
            resnet50_probes = self.train_and_eval_probes(tr_50_mean, train_labels, v_50_mean, val_labels)
            
        backbone_results = {
            "learned_scratch_resnet18": scratch_probes,
            "learned_pretrained_resnet18": ablation_results["Mean Pooled Spatial (512-dim)"],
            "raw_pretrained_resnet18": raw_probes,
            "resnet50": resnet50_probes if resnet50_available else "Skipped — pretrained ResNet-50 unavailable locally."
        }
        
        # Save backbone ablation results separately
        with open(os.path.join(self.output_dir, "backbone_ablation_results.json"), "w") as f:
            json.dump(backbone_results, f, indent=4)
            
        # Write flat diagnostic metrics
        diagnostic_metrics = {
            "ablation": ablation_results,
            "frame_count": frame_count_results,
            "backbone": backbone_results
        }
        
        with open(os.path.join(self.output_dir, "diagnostic_metrics.json"), "w") as f:
            json.dump(diagnostic_metrics, f, indent=4)
            
        # Save a flat CSV of all validation probe performances
        csv_path = os.path.join(self.output_dir, "diagnostic_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Configuration", "Probe Classifier", "Accuracy", "Macro-F1", "REAL Recall", "FAKE Recall", "ROC-AUC"])
            
            for name, clfs in ablation_results.items():
                for probe_name, metrics in clfs.items():
                    writer.writerow([
                        "Temporal Ablation", name, probe_name,
                        f"{metrics['accuracy']:.4f}", f"{metrics['macro_f1']:.4f}",
                        f"{metrics['real_recall']:.4f}", f"{metrics['fake_recall']:.4f}", f"{metrics['roc_auc']:.4f}"
                    ])
                    
            for name, clfs in frame_count_results.items():
                for probe_name, metrics in clfs.items():
                    writer.writerow([
                        "Frame Count", name, probe_name,
                        f"{metrics['accuracy']:.4f}", f"{metrics['macro_f1']:.4f}",
                        f"{metrics['real_recall']:.4f}", f"{metrics['fake_recall']:.4f}", f"{metrics['roc_auc']:.4f}"
                    ])
                    
            for name, val in backbone_results.items():
                if isinstance(val, str):
                    writer.writerow(["Backbone", name, "None", "0.0000", "0.0000", "0.0000", "0.0000", "0.5000"])
                    continue
                for probe_name, metrics in val.items():
                    writer.writerow([
                        "Backbone", name, probe_name,
                        f"{metrics['accuracy']:.4f}", f"{metrics['macro_f1']:.4f}",
                        f"{metrics['real_recall']:.4f}", f"{metrics['fake_recall']:.4f}", f"{metrics['roc_auc']:.4f}"
                    ])
                    
        logger.info(f"Diagnostics complete. Metrics saved to {self.output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default=os.getcwd())
    args = parser.parse_args()
    
    pipeline = DiagnosticsPipeline(args.workspace)
    pipeline.run_diagnostics()
