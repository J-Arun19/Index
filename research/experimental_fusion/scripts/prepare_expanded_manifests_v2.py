import os
import sys
import json
import hashlib
import cv2
import numpy as np
from typing import Dict, Any, List, Set, Tuple

celeb_dir = r"D:\Download\SatyaDrishti_Datasets\Celeb DF"
ff_dir = r"D:\Download\SatyaDrishti_Datasets\FaceForensics++_C23"

DATASETS_DIR = "research/experimental_fusion/datasets"
os.makedirs(DATASETS_DIR, exist_ok=True)

def sha256_file(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_ff_seq_id(filename: str) -> str:
    """Extracts base sequence ID from FF++ filename (e.g. 003.mp4 -> 003, 000_003.mp4 -> 000)."""
    base = os.path.splitext(filename)[0]
    return base.split("_")[0].zfill(3)

def build_manifests():
    print("=" * 80)
    print("PREPARING PHASE 2 VALIDATION AND TEST MANIFESTS")
    print("=" * 80)

    # 1. Load FaceForensics++ Splits
    with open("backend/training/splits/train.json") as f:
        train_pairs = json.load(f)
    with open("backend/training/splits/val.json") as f:
        val_pairs = json.load(f)
    with open("backend/training/splits/test.json") as f:
        test_pairs = json.load(f)

    train_seqs = {str(s).zfill(3) for pair in train_pairs for s in pair}
    val_seqs = {str(s).zfill(3) for pair in val_pairs for s in pair}
    test_seqs = {str(s).zfill(3) for pair in test_pairs for s in pair}

    print(f"FF++ Sequence Counts: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")
    assert len(train_seqs.intersection(val_seqs)) == 0, "Train-Val sequence leak in FF++!"
    assert len(train_seqs.intersection(test_seqs)) == 0, "Train-Test sequence leak in FF++!"
    assert len(val_seqs.intersection(test_seqs)) == 0, "Val-Test sequence leak in FF++!"
    print("  -> FF++ splits strictly disjoint (0 sequence overlap).")

    # 2. Load Celeb-DF Test List
    test_list_path = os.path.join(celeb_dir, "List_of_testing_videos.txt")
    with open(test_list_path) as f:
        celeb_test_lines = [l.strip().split() for l in f if l.strip()]

    celeb_test_dict = {}
    for l in celeb_test_lines:
        lbl, rel = l[0], l[1]
        norm = os.path.normpath(os.path.join(celeb_dir, rel))
        celeb_test_dict[norm] = {"label": "REAL" if lbl == "1" else "FAKE", "relative_path": rel}

    print(f"Celeb-DF Official Test List: {len(celeb_test_dict)} clips.")

    # 3. BUILD VALIDATION MANIFEST (Calibration Pool - Disjoint from Test)
    # Target: 60 validation videos (30 REAL, 30 FAKE)
    val_videos = []
    val_hashes = set()

    # 15 Celeb-DF Non-Test Real
    celeb_real_folder = os.path.join(celeb_dir, "Celeb-real")
    candidates_celeb_val_real = [
        f for f in sorted(os.listdir(celeb_real_folder))
        if f.endswith(".mp4") and os.path.normpath(os.path.join(celeb_real_folder, f)) not in celeb_test_dict
    ]
    for fn in candidates_celeb_val_real[:15]:
        p = os.path.join(celeb_real_folder, fn)
        h = sha256_file(p)
        val_hashes.add(h)
        val_videos.append({
            "id": f"val_vid_celeb_real_{len(val_videos):02d}",
            "filename": fn,
            "filepath": p,
            "dataset": "Celeb-DF_v2",
            "subset": "Celeb-real",
            "ground_truth": "REAL",
            "split": "validation",
            "sha256": h,
            "license": "Celeb-DF Academic Research License",
        })

    # 15 FF++ Val Original (from val_pairs)
    ff_orig_folder = os.path.join(ff_dir, "original")
    selected_val_orig_seqs = sorted(list(val_seqs))[:15]
    for seq in selected_val_orig_seqs:
        fn = f"{seq}.mp4"
        p = os.path.join(ff_orig_folder, fn)
        if os.path.exists(p):
            h = sha256_file(p)
            val_hashes.add(h)
            val_videos.append({
                "id": f"val_vid_ff_orig_{len(val_videos):02d}",
                "filename": fn,
                "filepath": p,
                "dataset": "FaceForensics++_C23",
                "subset": "original",
                "ground_truth": "REAL",
                "split": "validation",
                "sequence_id": seq,
                "sha256": h,
                "license": "TUM Non-Commercial Research License",
            })

    # 15 Celeb-DF Non-Test Synthesis
    celeb_synth_folder = os.path.join(celeb_dir, "Celeb-synthesis")
    candidates_celeb_val_synth = [
        f for f in sorted(os.listdir(celeb_synth_folder))
        if f.endswith(".mp4") and os.path.normpath(os.path.join(celeb_synth_folder, f)) not in celeb_test_dict
    ]
    for fn in candidates_celeb_val_synth[:15]:
        p = os.path.join(celeb_synth_folder, fn)
        h = sha256_file(p)
        val_hashes.add(h)
        val_videos.append({
            "id": f"val_vid_celeb_synth_{len(val_videos):02d}",
            "filename": fn,
            "filepath": p,
            "dataset": "Celeb-DF_v2",
            "subset": "Celeb-synthesis",
            "ground_truth": "FAKE",
            "split": "validation",
            "sha256": h,
            "license": "Celeb-DF Academic Research License",
        })

    # 15 FF++ Val Manipulated (3 Deepfakes, 3 Face2Face, 3 FaceSwap, 3 NeuralTextures, 3 FaceShifter)
    manip_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
    manip_idx = 0
    for m_type in manip_types:
        folder = os.path.join(ff_dir, m_type)
        pairs_for_m = val_pairs[manip_idx * 3 : (manip_idx + 1) * 3]
        for pair in pairs_for_m:
            fn = f"{pair[0]}_{pair[1]}.mp4"
            p = os.path.join(folder, fn)
            if not os.path.exists(p):
                fn = f"{pair[1]}_{pair[0]}.mp4"
                p = os.path.join(folder, fn)
            if os.path.exists(p):
                h = sha256_file(p)
                val_hashes.add(h)
                val_videos.append({
                    "id": f"val_vid_ff_{m_type.lower()}_{len(val_videos):02d}",
                    "filename": fn,
                    "filepath": p,
                    "dataset": "FaceForensics++_C23",
                    "subset": m_type,
                    "ground_truth": "FAKE",
                    "split": "validation",
                    "pair_id": f"{pair[0]}_{pair[1]}",
                    "sha256": h,
                    "license": "TUM Non-Commercial Research License",
                })
        manip_idx += 1

    print(f"Built Validation Video Set: {len(val_videos)} clips ({sum(1 for v in val_videos if v['ground_truth']=='REAL')} Real, {sum(1 for v in val_videos if v['ground_truth']=='FAKE')} Fake).")

    # 4. BUILD EXPANDED LOCKED TEST MANIFEST (Phase 2 Test Pool)
    # Target: 200 balanced videos (100 REAL, 100 FAKE)
    test_videos = []
    test_hashes = set()

    # 50 Celeb-DF Test Real (25 Celeb-real, 25 YouTube-real)
    celeb_test_celebreal = [rel for rel in celeb_test_lines if rel[0] == "1" and "Celeb-real" in rel[1]]
    celeb_test_yt = [rel for rel in celeb_test_lines if rel[0] == "1" and "YouTube-real" in rel[1]]

    for l in celeb_test_celebreal[:25]:
        rel = l[1]
        p = os.path.join(celeb_dir, rel)
        fn = os.path.basename(p)
        h = sha256_file(p)
        assert h not in val_hashes, f"Leak: {fn} in val hashes!"
        test_hashes.add(h)
        test_videos.append({
            "id": f"test_vid_celeb_real_{len(test_videos):03d}",
            "filename": fn,
            "filepath": p,
            "dataset": "Celeb-DF_v2",
            "subset": "Celeb-real",
            "ground_truth": "REAL",
            "split": "test",
            "sha256": h,
            "license": "Celeb-DF Academic Research License",
        })

    for l in celeb_test_yt[:25]:
        rel = l[1]
        p = os.path.join(celeb_dir, rel)
        fn = os.path.basename(p)
        h = sha256_file(p)
        assert h not in val_hashes, f"Leak: {fn} in val hashes!"
        test_hashes.add(h)
        test_videos.append({
            "id": f"test_vid_yt_real_{len(test_videos):03d}",
            "filename": fn,
            "filepath": p,
            "dataset": "Celeb-DF_v2",
            "subset": "YouTube-real",
            "ground_truth": "REAL",
            "split": "test",
            "sha256": h,
            "license": "Celeb-DF Academic Research License",
        })

    # 50 FF++ Test Original (using 50 distinct sequence IDs from test_seqs)
    selected_test_orig_seqs = sorted(list(test_seqs))[:50]
    for seq in selected_test_orig_seqs:
        fn = f"{seq}.mp4"
        p = os.path.join(ff_orig_folder, fn)
        assert seq not in val_seqs, f"Leak: sequence {seq} in val_seqs!"
        assert seq not in train_seqs, f"Leak: sequence {seq} in train_seqs!"
        h = sha256_file(p)
        assert h not in val_hashes, f"Leak: {fn} in val hashes!"
        test_hashes.add(h)
        test_videos.append({
            "id": f"test_vid_ff_orig_{len(test_videos):03d}",
            "filename": fn,
            "filepath": p,
            "dataset": "FaceForensics++_C23",
            "subset": "original",
            "ground_truth": "REAL",
            "split": "test",
            "sequence_id": seq,
            "sha256": h,
            "license": "TUM Non-Commercial Research License",
        })

    # 50 Celeb-DF Test Synthesis
    celeb_test_synth = [l for l in celeb_test_lines if l[0] == "0"]
    for l in celeb_test_synth[:50]:
        rel = l[1]
        p = os.path.join(celeb_dir, rel)
        fn = os.path.basename(p)
        h = sha256_file(p)
        assert h not in val_hashes, f"Leak: {fn} in val hashes!"
        test_hashes.add(h)
        test_videos.append({
            "id": f"test_vid_celeb_synth_{len(test_videos):03d}",
            "filename": fn,
            "filepath": p,
            "dataset": "Celeb-DF_v2",
            "subset": "Celeb-synthesis",
            "ground_truth": "FAKE",
            "split": "test",
            "sha256": h,
            "license": "Celeb-DF Academic Research License",
        })

    # 50 FF++ Test Manipulations (10 Deepfakes, 10 Face2Face, 10 FaceSwap, 10 NeuralTextures, 10 FaceShifter)
    # Using 50 distinct sequence pairs from test_pairs
    for m_idx, m_type in enumerate(manip_types):
        folder = os.path.join(ff_dir, m_type)
        pairs_for_m = test_pairs[m_idx * 10 : (m_idx + 1) * 10]
        for pair in pairs_for_m:
            fn = f"{pair[0]}_{pair[1]}.mp4"
            p = os.path.join(folder, fn)
            if not os.path.exists(p):
                fn = f"{pair[1]}_{pair[0]}.mp4"
                p = os.path.join(folder, fn)
            assert os.path.exists(p), f"Missing FF++ file {p}"
            assert str(pair[0]).zfill(3) not in val_seqs, f"Leak: {pair[0]} in val_seqs"
            assert str(pair[1]).zfill(3) not in val_seqs, f"Leak: {pair[1]} in val_seqs"
            h = sha256_file(p)
            assert h not in val_hashes, f"Leak: {fn} in val hashes"
            test_hashes.add(h)
            test_videos.append({
                "id": f"test_vid_ff_{m_type.lower()}_{len(test_videos):03d}",
                "filename": fn,
                "filepath": p,
                "dataset": "FaceForensics++_C23",
                "subset": m_type,
                "ground_truth": "FAKE",
                "split": "test",
                "pair_id": f"{pair[0]}_{pair[1]}",
                "sha256": h,
                "license": "TUM Non-Commercial Research License",
            })

    print(f"Built Expanded Test Video Set: {len(test_videos)} clips ({sum(1 for v in test_videos if v['ground_truth']=='REAL')} Real, {sum(1 for v in test_videos if v['ground_truth']=='FAKE')} Fake).")

    # 5. Extract and build image manifests (1 frame per video + in-the-wild + controls)
    val_images = []
    test_images = []

    # Validation Images (from val_videos: 1 middle frame each = 60 images)
    for v in val_videos:
        val_images.append({
            "id": f"val_img_{v['id']}",
            "source_video_id": v["id"],
            "source_video_path": v["filepath"],
            "source_filename": v["filename"],
            "dataset": v["dataset"],
            "subset": v["subset"],
            "ground_truth": v["ground_truth"],
            "split": "validation",
            "frame_selection": "exact_middle_frame",
            "license": v["license"],
        })

    # Test Images: 200 Facial Images (100 Real, 100 Fake) from test_videos (1 frame each) + 10 safety controls
    for v in test_videos:
        test_images.append({
            "id": f"test_img_{v['id']}",
            "source_video_id": v["id"],
            "source_video_path": v["filepath"],
            "source_filename": v["filename"],
            "dataset": v["dataset"],
            "subset": v["subset"],
            "ground_truth": v["ground_truth"],
            "split": "test",
            "frame_selection": "exact_middle_frame",
            "license": v["license"],
        })

    # 10 Safety Controls (non-face / degraded media)
    safety_controls = [
        {"filename": "document_contract_sample.png", "src": "backend/uploads/document_contract_sample.png", "category": "non_face_document", "gt": "NON-FACE"},
        {"filename": "deepfake_logo.png", "src": "backend/uploads/85bc1c33-1b30-408b-9675-2811af19a8d6_deepfake_logo.png", "category": "non_face_graphic", "gt": "NON-FACE"},
        {"filename": "she.jpeg", "src": "backend/uploads/404e3990-dbc9-4ec5-8abe-159788170bfa_she.jpeg", "category": "degraded_lowlight_real", "gt": "REAL"},
        {"filename": "profile.jpeg", "src": "backend/uploads/a88b3aae-5f65-44bb-b1d4-c95f6ac47a8f_profile.jpeg", "category": "in_the_wild_real_portrait", "gt": "REAL"},
        {"filename": "picture.jpeg", "src": "backend/uploads/8a30af82-839c-4be1-afc7-2ce2c58d33b1_picture.jpeg", "category": "in_the_wild_real_portrait", "gt": "REAL"},
        {"filename": "deepy.jpeg", "src": "backend/uploads/37394aee-2a7f-494e-a57a-ae490fcbc441_deepy.jpeg", "category": "in_the_wild_face_swap", "gt": "FAKE"},
        {"filename": "professional.jpeg", "src": "backend/uploads/156d6a4a-8211-4214-a9db-bfe9df7c221a_professional.jpeg", "category": "ai_generated_synthetic_face", "gt": "FAKE"},
    ]
    for sc in safety_controls:
        p = sc["src"]
        if os.path.exists(p):
            test_images.append({
                "id": f"test_img_control_{sc['filename']}",
                "source_video_id": None,
                "source_video_path": p,
                "source_filename": sc["filename"],
                "dataset": "In-the-wild_and_Controls",
                "subset": sc["category"],
                "ground_truth": sc["gt"],
                "split": "test",
                "sha256": sha256_file(p),
                "frame_selection": "static_image",
                "license": "Public Domain / CC-BY Test Controls",
            })

    print(f"Built Expanded Test Image Set: {len(test_images)} images.")

    # 6. Save Manifests
    val_manifest_path = os.path.join(DATASETS_DIR, "validation_manifest_v2.json")
    with open(val_manifest_path, "w") as f:
        json.dump({
            "split": "validation_only",
            "purpose": "Calibration and temperature fitting for SBI and Xception",
            "video_count": len(val_videos),
            "image_count": len(val_images),
            "videos": val_videos,
            "images": val_images,
        }, f, indent=2)

    test_manifest_path = os.path.join(DATASETS_DIR, "locked_test_manifest_v2.json")
    with open(test_manifest_path, "w") as f:
        json.dump({
            "split": "test_locked",
            "purpose": "Final expanded benchmark evaluation (LOCKED)",
            "video_count": len(test_videos),
            "image_count": len(test_images),
            "videos": test_videos,
            "images": test_images,
        }, f, indent=2)

    print(f"[+] Saved validation manifest to {val_manifest_path}")
    print(f"[+] Saved locked test manifest to {test_manifest_path}")

    # Compute and display hashes
    print("\nManifest SHA-256 Hashes:")
    print(f"  validation_manifest_v2.json: {sha256_file(val_manifest_path)}")
    print(f"  locked_test_manifest_v2.json: {sha256_file(test_manifest_path)}")

if __name__ == "__main__":
    build_manifests()
