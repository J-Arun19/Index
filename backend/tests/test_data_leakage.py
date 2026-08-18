import os
import csv
import re
import unittest
from backend.utils.config import PipelineConfig

class TestDataLeakage(unittest.TestCase):
    """Automated unit tests to verify zero data leakage across train, val, and test splits."""

    @classmethod
    def setUpClass(cls):
        cls.manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
        cls.manifest_exists = os.path.exists(cls.manifest_path)

    def test_manifest_existence(self):
        """Ensure the dataset manifest exists before running split checks."""
        self.assertTrue(self.manifest_exists, f"Manifest not found at {self.manifest_path}. Please run generate_manifest.py first.")

    def test_file_path_exclusivity(self):
        """Ensure no physical file or video path is shared across splits."""
        if not self.manifest_exists:
            self.skipTest("Manifest file missing.")

        splits_by_path = {}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.normpath(row["file_path"])
                split = row["split"].lower()
                
                if path in splits_by_path:
                    # File path already registered, it must match the same split
                    self.assertEqual(
                        splits_by_path[path], split,
                        f"Leakage detected! Path '{path}' appears in multiple splits: '{splits_by_path[path]}' and '{split}'."
                    )
                else:
                    splits_by_path[path] = split

    def test_video_derived_frames_split_match(self):
        """Ensure frames extracted from a video inherit the exact same split as the video itself."""
        if not self.manifest_exists:
            self.skipTest("Manifest file missing.")

        # Map video file paths to their splits
        video_splits = {}
        image_frames = []

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = os.path.normpath(row["file_path"])
                media_type = row["media_type"].lower()
                split = row["split"].lower()
                origin = row.get("sample_origin", "").lower()
                
                if media_type == "video":
                    video_splits[path] = split
                elif origin == "video_frame":
                    image_frames.append((path, split))

        # Check all video-derived image frames match their source video split
        for path, img_split in image_frames:
            if path in video_splits:
                self.assertEqual(
                    video_splits[path], img_split,
                    f"Leakage detected! Derived frame for '{path}' has split '{img_split}', but source video has split '{video_splits[path]}'."
                )

    def test_celebdf_identity_exclusivity(self):
        """Ensure subject identities in Celeb-DF do not leak between splits (checking all IDs in filename)."""
        if not self.manifest_exists:
            self.skipTest("Manifest file missing.")

        split_identities = {"train": set(), "val": set(), "test": set()}

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["dataset_source"] != "Celeb_DF":
                    continue
                
                filename = os.path.basename(row["file_path"])
                # Extract ALL subject IDs from the filename (e.g. id0_id16_0000.mp4 -> [id0, id16])
                subj_ids = re.findall(r'(id\d+)', filename)
                for subj_id in subj_ids:
                    split = row["split"].lower()
                    if split in split_identities:
                        split_identities[split].add(subj_id)

        train_val_overlap = split_identities["train"].intersection(split_identities["val"])
        train_test_overlap = split_identities["train"].intersection(split_identities["test"])
        val_test_overlap = split_identities["val"].intersection(split_identities["test"])

        self.assertEqual(len(train_val_overlap), 0, f"Celeb-DF identity leak between train and val: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"Celeb-DF identity leak between train and test: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"Celeb-DF identity leak between val and test: {val_test_overlap}")

    def test_faceforensics_sequence_exclusivity(self):
        """Ensure FaceForensics++ sequences do not leak between splits."""
        if not self.manifest_exists:
            self.skipTest("Manifest file missing.")

        split_sequences = {"train": set(), "val": set(), "test": set()}

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["dataset_source"] not in ("FaceForensics++_C23", "Faceforensic++"):
                    continue
                # Skip DeepFakeDetection as it has a dedicated actor-based test
                if row["manipulation_type"] == "DeepFakeDetection":
                    continue
                
                filename = os.path.basename(row["file_path"])
                m = re.match(r'(\d+)', filename)
                if m:
                    seq_id = m.group(1).zfill(3)
                else:
                    parts = row["file_path"].split(os.sep)
                    seq_id = None
                    for part in parts:
                        if part.isdigit() and len(part) <= 4:
                            seq_id = part.zfill(3)
                            break
                
                if seq_id:
                    split = row["split"].lower()
                    if split in split_sequences:
                        split_sequences[split].add(seq_id)

        train_val_overlap = split_sequences["train"].intersection(split_sequences["val"])
        train_test_overlap = split_sequences["train"].intersection(split_sequences["test"])
        val_test_overlap = split_sequences["val"].intersection(split_sequences["test"])

        self.assertEqual(len(train_val_overlap), 0, f"FaceForensics sequence leak between train and val: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"FaceForensics sequence leak between train and test: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"FaceForensics sequence leak between val and test: {val_test_overlap}")

    def test_dfd_actor_exclusivity(self):
        """Ensure actor identities in DeepFakeDetection do not leak between splits."""
        if not self.manifest_exists:
            self.skipTest("Manifest file missing.")

        split_actors = {"train": set(), "val": set(), "test": set()}

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["dataset_source"] != "FaceForensics++_C23" or row["manipulation_type"] != "DeepFakeDetection":
                    continue
                
                filename = os.path.basename(row["file_path"])
                m = re.match(r'(\d+)_(\d+)', filename)
                if m:
                    a1, a2 = m.groups()
                    split = row["split"].lower()
                    if split in split_actors:
                        split_actors[split].add(a1)
                        split_actors[split].add(a2)

        train_val_overlap = split_actors["train"].intersection(split_actors["val"])
        train_test_overlap = split_actors["train"].intersection(split_actors["test"])
        val_test_overlap = split_actors["val"].intersection(split_actors["test"])

        self.assertEqual(len(train_val_overlap), 0, f"DFD actor leak between train and val: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"DFD actor leak between train and test: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"DFD actor leak between val and test: {val_test_overlap}")

if __name__ == "__main__":
    unittest.main()
