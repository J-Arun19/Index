import csv
import os
import re

manifest_path = r"d:\Download\satya-vision-guard-main\satya-vision-guard-main\backend\training\dataset_manifest.csv"

def get_seq_id(row):
    if row["dataset_source"] not in ("FaceForensics++_C23", "Faceforensic++"):
        return None
    filename = os.path.basename(row["file_path"])
    m = re.match(r'(\d+)', filename)
    if m:
        return m.group(1).zfill(3)
    return None

seq_splits = {}
with open(manifest_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        seq = get_seq_id(row)
        if seq:
            split = row["split"]
            if seq not in seq_splits:
                seq_splits[seq] = {}
            if split not in seq_splits[seq]:
                seq_splits[seq][split] = []
            seq_splits[seq][split].append(i)

for seq, splits in seq_splits.items():
    if len(splits) > 1:
        print(f"Sequence {seq} is in multiple splits:")
        for split, rows in splits.items():
            print(f"  {split}: {len(rows)} rows, first row is {rows[0]}")
            # let's print the first row's path
            with open(manifest_path, "r", encoding="utf-8") as f2:
                r2 = list(csv.DictReader(f2))[rows[0]]
                print(f"    Path: {r2['file_path']}")
