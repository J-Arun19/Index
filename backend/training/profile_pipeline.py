import os
import sys
import time
import torch
import numpy as np
import cv2
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.training.dataset_loader import DeepfakeDataset
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.preprocessing.face_detector import FaceDetector

logger = setup_logger("PipelineProfiler")

def profile_individual_samples(num_samples: int = 100):
    logger.info("==================================================")
    logger.info(f"  PROFILING INDIVIDUAL SAMPLE PREPROCESSING ({num_samples} samples)  ")
    logger.info("==================================================")
    
    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
    
    t_video_extract = []
    t_face_detect = []
    t_resize_color = []
    t_aug_norm = []
    t_total_sample = []
    
    detector = FaceDetector()
    preprocessor = ImagePreprocessor()
    
    sample_indices = np.random.choice(len(dataset.samples), size=min(num_samples, len(dataset.samples)), replace=False)
    
    for count, idx in enumerate(sample_indices):
        sample = dataset.samples[idx]
        file_path = sample["file_path"]
        origin = sample["sample_origin"]
        frame_idx = sample["frame_index"]
        
        t0 = time.perf_counter()
        
        # 1. Video frame extraction / image load
        t1_start = time.perf_counter()
        if origin == "video_frame":
            cap = None
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        else:
            frame = cv2.imread(file_path)
        t1_end = time.perf_counter()
        dur_extract = (t1_end - t1_start) * 1000.0  # ms
        
        if frame is None:
            continue
            
        # 2. Haar Cascade Face Detection
        t2_start = time.perf_counter()
        faces = detector.detect_faces(frame)
        if faces:
            frame_cropped = detector.crop_face(frame, faces[0])
        else:
            frame_cropped = frame
        t2_end = time.perf_counter()
        dur_detect = (t2_end - t2_start) * 1000.0  # ms
        
        # 3. Resize to 224x224 & BGR to RGB
        t3_start = time.perf_counter()
        resized = cv2.resize(frame_cropped, (224, 224))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        t3_end = time.perf_counter()
        dur_resize = (t3_end - t3_start) * 1000.0  # ms
        
        # 4. Augmentation & Normalization to Tensor
        t4_start = time.perf_counter()
        tensor = preprocessor.apply_augmentation_and_normalization(img_rgb, is_training=True)
        t4_end = time.perf_counter()
        dur_aug_norm = (t4_end - t4_start) * 1000.0  # ms
        
        dur_total = (time.perf_counter() - t0) * 1000.0
        
        t_video_extract.append(dur_extract)
        t_face_detect.append(dur_detect)
        t_resize_color.append(dur_resize)
        t_aug_norm.append(dur_aug_norm)
        t_total_sample.append(dur_total)
        
    avg_extract = np.mean(t_video_extract) if t_video_extract else 0
    avg_detect = np.mean(t_face_detect) if t_face_detect else 0
    avg_resize = np.mean(t_resize_color) if t_resize_color else 0
    avg_aug_norm = np.mean(t_aug_norm) if t_aug_norm else 0
    avg_total = np.mean(t_total_sample) if t_total_sample else 0
    
    print("\n" + "="*60)
    print("INPUT DATA PIPELINE MICRO-PROFILING RESULTS (PER SAMPLE):")
    print("="*60)
    print(f"1. Video Frame Extraction / Read: {avg_extract:.2f} ms ({avg_extract/max(1e-5, avg_total)*100:.1f}%)")
    print(f"2. Haar Face Detection (CPU):    {avg_detect:.2f} ms ({avg_detect/max(1e-5, avg_total)*100:.1f}%)")
    print(f"3. Resize (224x224) & BGR->RGB:  {avg_resize:.2f} ms ({avg_resize/max(1e-5, avg_total)*100:.1f}%)")
    print(f"4. Augmentation & Normalization: {avg_aug_norm:.2f} ms ({avg_aug_norm/max(1e-5, avg_total)*100:.1f}%)")
    print("-" * 60)
    print(f"TOTAL CPU PROCESSING TIME:       {avg_total:.2f} ms per sample")
    print(f"Theoretical Max Single-Core CPU: {1000.0/max(1e-5, avg_total):.2f} samples/sec")
    print("="*60 + "\n")

if __name__ == "__main__":
    profile_individual_samples(100)
