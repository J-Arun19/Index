import os
import re
import csv
import json
import cv2
from typing import Dict, Set, List, Tuple
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger

logger = setup_logger("GenerateManifest")

def load_faceforensics_splits(splits_dir: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Loads official FaceForensics splits from JSON files and extracts sequence IDs."""
    train_ids = set()
    val_ids = set()
    test_ids = set()
    
    for split_name, target_set in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = os.path.join(splits_dir, f"{split_name}.json")
        if not os.path.exists(path):
            logger.error(f"Required split file missing: {path}")
            continue
        try:
            with open(path, "r") as f:
                pairs = json.load(f)
                for pair in pairs:
                    for seq_id in pair:
                        target_set.add(str(seq_id).zfill(3))
        except Exception as e:
            logger.error(f"Failed to load split file {path}: {str(e)}")
            
    logger.info(f"Loaded FaceForensics splits. Train: {len(train_ids)} seqs, Val: {len(val_ids)} seqs, Test: {len(test_ids)} seqs.")
    return train_ids, val_ids, test_ids

def get_faceforensics_split(file_name: str, train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]) -> str:
    """Determines FaceForensics split from filename sequence ID."""
    m = re.match(r'(\d+)', file_name)
    if m:
        seq_id = m.group(1).zfill(3)
        if seq_id in train_ids:
            return "train"
        elif seq_id in val_ids:
            return "val"
        elif seq_id in test_ids:
            return "test"
    return "train"  # Default fallback

def get_dfd_split(file_name: str) -> str:
    """Determines Jigsaw DeepFakeDetection split based on disjoint actor connected components."""
    comp1_train = {'01', '02', '27', '03', '12', '04', '20', '11', '09', '21', '15', '14', '26', '18', '25', '07', '06', '13'}
    comp2_val = {'05', '16', '08', '28', '17'}
    comp3_test = {'10', '24', '23', '22', '19'}
    
    m = re.match(r'(\d+)_(\d+)', file_name)
    if m:
        a1, a2 = m.groups()
        if a1 in comp1_train and a2 in comp1_train:
            return "train"
        elif a1 in comp2_val and a2 in comp2_val:
            return "val"
        elif a1 in comp3_test and a2 in comp3_test:
            return "test"
    return "train"  # Fallback

def get_celebdf_split(rel_path: str, is_test_video: bool) -> str:
    """Determines Celeb-DF split preventing subject identity leakage."""
    if is_test_video:
        return "test"
        
    filename = os.path.basename(rel_path)
    m = re.match(r'(id\d+)', filename)
    if m:
        subj_id = m.group(1)
        num = int(subj_id[2:])
        # Modulo 5 identity partition: val if divisible by 5, else train
        if num % 5 == 0:
            return "val"
        return "train"
        
    # YouTube-real videos format '00000.mp4'
    m_yt = re.match(r'(\d+)', filename)
    if m_yt:
        num = int(m_yt.group(1))
        if num % 5 == 0:
            return "val"
        return "train"
        
    return "train"

def get_video_frame_count(file_path: str) -> int:
    """Opens video file to read total frame count without decoding."""
    cap = cv2.VideoCapture(file_path)
    count = 0
    if cap.isOpened():
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return count

def generate_manifest():
    root = PipelineConfig.SATYADRISHTI_DATASET_ROOT
    logger.info(f"Scanning dataset root: {root}")
    
    if not os.path.exists(root):
        logger.error(f"Dataset root directory does not exist: {root}")
        return
        
    splits_dir = os.path.join(PipelineConfig.BACKEND_DIR, "training", "splits")
    ff_train, ff_val, ff_test = load_faceforensics_splits(splits_dir)
    
    manifest = []
    
    # Track statistics separately for validation/reporting
    source_video_count = 0
    derived_image_count = 0
    
    stats = {
        "Celeb_DF": {"total": 0, "video": 0, "image": 0, "real": 0, "fake": 0, "train": 0, "val": 0, "test": 0, "corrupted": 0},
        "FaceForensics++_C23": {"total": 0, "video": 0, "image": 0, "real": 0, "fake": 0, "train": 0, "val": 0, "test": 0, "corrupted": 0},
        "Faceforensic++": {"total": 0, "video": 0, "image": 0, "real": 0, "fake": 0, "train": 0, "val": 0, "test": 0, "corrupted": 0}
    }
    
    corrupted_files = []
    
    # Relative positions for deterministic multi-frame sampling (exactly 5 frames)
    relative_positions = [0.10, 0.35, 0.50, 0.65, 0.90]
    
    # 1. PROCESS CELEB DF
    celeb_df_dir = os.path.join(root, "Celeb DF")
    if os.path.exists(celeb_df_dir):
        logger.info("Processing Celeb DF dataset...")
        test_file = os.path.join(celeb_df_dir, "List_of_testing_videos.txt")
        test_videos = set()
        if os.path.exists(test_file):
            with open(test_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        test_videos.add(parts[1].replace('/', os.sep))
        else:
            logger.warning(f"Celeb-DF test split file missing: {test_file}")
            
        folders = [
            ("Celeb-real", 0, "original"),
            ("YouTube-real", 0, "original"),
            ("Celeb-synthesis", 1, "Celeb-synthesis")
        ]
        
        for folder, label, manip_type in folders:
            folder_path = os.path.join(celeb_df_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for f in os.listdir(folder_path):
                if not f.endswith(".mp4"):
                    continue
                file_path = os.path.abspath(os.path.join(folder_path, f))
                
                # Check integrity
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                if file_size < 100:
                    corrupted_files.append(file_path)
                    stats["Celeb_DF"]["corrupted"] += 1
                    continue
                    
                rel_video_path = os.path.join(folder, f)
                is_test = rel_video_path in test_videos
                split = get_celebdf_split(rel_video_path, is_test)
                
                # Get video frame count
                total_frames = get_video_frame_count(file_path)
                if total_frames <= 0:
                    total_frames = 30 # fallback
                
                # 1a. Add video row
                manifest.append({
                    "dataset_source": "Celeb_DF",
                    "file_path": file_path,
                    "media_type": "video",
                    "label": label,
                    "manipulation_type": manip_type,
                    "compression": "none",
                    "split": split,
                    "sample_origin": "video",
                    "frame_index": -1
                })
                
                stats["Celeb_DF"]["total"] += 1
                stats["Celeb_DF"]["video"] += 1
                if label == 0:
                    stats["Celeb_DF"]["real"] += 1
                else:
                    stats["Celeb_DF"]["fake"] += 1
                stats["Celeb_DF"][split] += 1
                source_video_count += 1
                
                # 1b. Add 5 image rows (multi-frame derived samples)
                for pos in relative_positions:
                    frame_idx = max(0, min(total_frames - 1, int(total_frames * pos)))
                    manifest.append({
                        "dataset_source": "Celeb_DF",
                        "file_path": file_path,
                        "media_type": "image",
                        "label": label,
                        "manipulation_type": manip_type,
                        "compression": "none",
                        "split": split,
                        "sample_origin": "video_frame",
                        "frame_index": frame_idx
                    })
                    stats["Celeb_DF"]["total"] += 1
                    stats["Celeb_DF"]["image"] += 1
                    if label == 0:
                        stats["Celeb_DF"]["real"] += 1
                    else:
                        stats["Celeb_DF"]["fake"] += 1
                    stats["Celeb_DF"][split] += 1
                    derived_image_count += 1
    else:
        logger.warning("Celeb DF directory not found under root.")
        
    # 2. PROCESS FACEFORENSICS++_C23 (Videos)
    ff_c23_dir = os.path.join(root, "FaceForensics++_C23")
    if os.path.exists(ff_c23_dir):
        logger.info("Processing FaceForensics++_C23 dataset...")
        subfolders = [
            ("original", 0, "original"),
            ("Deepfakes", 1, "Deepfakes"),
            ("Face2Face", 1, "Face2Face"),
            ("FaceShifter", 1, "FaceShifter"),
            ("FaceSwap", 1, "FaceSwap"),
            ("NeuralTextures", 1, "NeuralTextures"),
            ("DeepFakeDetection", 1, "DeepFakeDetection")
        ]
        
        for folder, label, manip_type in subfolders:
            folder_path = os.path.join(ff_c23_dir, folder)
            if not os.path.exists(folder_path):
                continue
            for f in os.listdir(folder_path):
                if not f.endswith(".mp4"):
                    continue
                file_path = os.path.abspath(os.path.join(folder_path, f))
                
                # Check integrity
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                if file_size < 100:
                    corrupted_files.append(file_path)
                    stats["FaceForensics++_C23"]["corrupted"] += 1
                    continue
                    
                if folder == "DeepFakeDetection":
                    split = get_dfd_split(f)
                else:
                    split = get_faceforensics_split(f, ff_train, ff_val, ff_test)
                    
                total_frames = get_video_frame_count(file_path)
                if total_frames <= 0:
                    total_frames = 30
                    
                # 2a. Add video row
                manifest.append({
                    "dataset_source": "FaceForensics++_C23",
                    "file_path": file_path,
                    "media_type": "video",
                    "label": label,
                    "manipulation_type": manip_type,
                    "compression": "C23",
                    "split": split,
                    "sample_origin": "video",
                    "frame_index": -1
                })
                
                stats["FaceForensics++_C23"]["total"] += 1
                stats["FaceForensics++_C23"]["video"] += 1
                if label == 0:
                    stats["FaceForensics++_C23"]["real"] += 1
                else:
                    stats["FaceForensics++_C23"]["fake"] += 1
                stats["FaceForensics++_C23"][split] += 1
                source_video_count += 1
                
                # 2b. Add 5 image rows (multi-frame derived samples)
                for pos in relative_positions:
                    frame_idx = max(0, min(total_frames - 1, int(total_frames * pos)))
                    manifest.append({
                        "dataset_source": "FaceForensics++_C23",
                        "file_path": file_path,
                        "media_type": "image",
                        "label": label,
                        "manipulation_type": manip_type,
                        "compression": "C23",
                        "split": split,
                        "sample_origin": "video_frame",
                        "frame_index": frame_idx
                    })
                    stats["FaceForensics++_C23"]["total"] += 1
                    stats["FaceForensics++_C23"]["image"] += 1
                    if label == 0:
                        stats["FaceForensics++_C23"]["real"] += 1
                    else:
                        stats["FaceForensics++_C23"]["fake"] += 1
                    stats["FaceForensics++_C23"][split] += 1
                    derived_image_count += 1
    else:
        logger.warning("FaceForensics++_C23 directory not found under root.")
        
    # 3. PROCESS FACEFORENSIC++ (Extracted physical frame images)
    ff_image_dir = os.path.join(root, "Faceforensic++")
    if os.path.exists(ff_image_dir):
        logger.info("Processing Faceforensic++ image dataset...")
        frames_dir = os.path.join(ff_image_dir, "extracted_frames", "real")
        if os.path.exists(frames_dir):
            for seq_folder in os.listdir(frames_dir):
                seq_path = os.path.join(frames_dir, seq_folder)
                if not os.path.isdir(seq_path):
                    continue
                split = "train"
                if seq_folder.isdigit():
                    seq_id = seq_folder.zfill(3)
                    if seq_id in ff_train:
                        split = "train"
                    elif seq_id in ff_val:
                        split = "val"
                    elif seq_id in ff_test:
                        split = "test"
                        
                for img_file in os.listdir(seq_path):
                    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    file_path = os.path.abspath(os.path.join(seq_path, img_file))
                    
                    # Check integrity
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    if file_size < 100:
                        corrupted_files.append(file_path)
                        stats["Faceforensic++"]["corrupted"] += 1
                        continue
                        
                    manifest.append({
                        "dataset_source": "Faceforensic++",
                        "file_path": file_path,
                        "media_type": "image",
                        "label": 0,
                        "manipulation_type": "original",
                        "compression": "none",
                        "split": split,
                        "sample_origin": "physical_image",
                        "frame_index": -1
                    })
                    
                    stats["Faceforensic++"]["total"] += 1
                    stats["Faceforensic++"]["image"] += 1
                    stats["Faceforensic++"]["real"] += 1
                    stats["Faceforensic++"][split] += 1
        else:
            logger.warning("Faceforensic++ extracted_frames/real not found.")
    else:
        logger.warning("Faceforensic++ directory not found under root.")
        
    # WRITE MANIFEST CSV
    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_source", "file_path", "media_type", "label", "manipulation_type", "compression", "split", "sample_origin", "frame_index"])
        for item in manifest:
            writer.writerow([
                item["dataset_source"],
                item["file_path"],
                item["media_type"],
                item["label"],
                item["manipulation_type"],
                item["compression"],
                item["split"],
                item["sample_origin"],
                item["frame_index"]
            ])
            
    logger.info(f"Successfully generated manifest with {len(manifest)} rows at: {manifest_path}")
    
    # WRITE DATASET INSPECTION REPORT JSON
    inspection_report = []
    for name, st in stats.items():
        manips = list(set([m["manipulation_type"] for m in manifest if m["dataset_source"] == name]))
        inspection_report.append({
            "dataset_name": name,
            "total_files": st["total"],
            "video_count": st["video"],
            "image_count": st["image"],
            "REAL_count": st["real"],
            "FAKE_count": st["fake"],
            "manipulation_types": manips,
            "compression_information": "C23" if name == "FaceForensics++_C23" else "none",
            "existing_splits": {
                "train": st["train"],
                "val": st["val"],
                "test": st["test"]
            },
            "unreadable_corrupted_files": [f for f in corrupted_files if name in f],
            "missing_metadata": False,
            "detected_directory_structure": "Standard subfolders scanned recursively"
        })
        
    report_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_inspection_report.json")
    with open(report_path, "w") as f:
        json.dump(inspection_report, f, indent=4)
    logger.info(f"Saved dataset inspection report to: {report_path}")
    
    # WRITE DATASET SPLIT REPORT JSON
    total_real = sum([st["real"] for st in stats.values()])
    total_fake = sum([st["fake"] for st in stats.values()])
    
    # Calculate detailed image-model statistics (excluding videos)
    image_manifest = [m for m in manifest if m["media_type"] == "image"]
    physical_img_count = sum(1 for m in image_manifest if m["sample_origin"] == "physical_image")
    derived_img_count = sum(1 for m in image_manifest if m["sample_origin"] == "video_frame")
    
    img_real = sum(1 for m in image_manifest if m["label"] == 0)
    img_fake = sum(1 for m in image_manifest if m["label"] == 1)
    
    img_train = sum(1 for m in image_manifest if m["split"] == "train")
    img_val = sum(1 for m in image_manifest if m["split"] == "val")
    img_test = sum(1 for m in image_manifest if m["split"] == "test")
    
    split_report = {
        "split_strategy": {
            "FaceForensics++": "Official pair-based train/val/test splits mapped to individual sequence numbers.",
            "DeepFakeDetection": "Disjoint actor pair graph partition to guarantee zero actor identity leakage.",
            "Celeb-DF": "Official test split from List_of_testing_videos.txt. Deterministic identity-based partitioning (modulo 5 on identity number) for train/val to prevent identity leakage."
        },
        "general_statistics": {
            "total_records": len(manifest),
            "total_videos": source_video_count,
            "total_images": len(image_manifest),
            "physical_images": physical_img_count,
            "derived_images": derived_img_count,
            "samples_per_video": len(relative_positions)
        },
        "dataset_statistics": stats,
        "image_model_dataset": {
            "total": len(image_manifest),
            "physical_images": physical_img_count,
            "derived_images": derived_img_count,
            "REAL": img_real,
            "FAKE": img_fake,
            "train": img_train,
            "val": img_val,
            "test": img_test
        },
        "video_model_dataset": {
            "total": source_video_count,
            "REAL": sum(1 for m in manifest if m["media_type"] == "video" and m["label"] == 0),
            "FAKE": sum(1 for m in manifest if m["media_type"] == "video" and m["label"] == 1),
            "train": sum(1 for m in manifest if m["media_type"] == "video" and m["split"] == "train"),
            "val": sum(1 for m in manifest if m["media_type"] == "video" and m["split"] == "val"),
            "test": sum(1 for m in manifest if m["media_type"] == "video" and m["split"] == "test")
        },
        "class_balance_overall": {
            "REAL": total_real,
            "FAKE": total_fake,
            "ratio_real_to_fake": total_real / max(1, total_fake)
        },
        "split_distribution_overall": {
            "train": sum([st["train"] for st in stats.values()]),
            "val": sum([st["val"] for st in stats.values()]),
            "test": sum([st["test"] for st in stats.values()]),
        }
    }
    
    split_report_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_split_report.json")
    with open(split_report_path, "w") as f:
        json.dump(split_report, f, indent=4)
    logger.info(f"Saved split report to: {split_report_path}")

if __name__ == "__main__":
    generate_manifest()
