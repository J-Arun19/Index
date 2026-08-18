import csv
import os

manifest_path = r"d:\Download\satya-vision-guard-main\satya-vision-guard-main\backend\training\dataset_manifest.csv"

with open(manifest_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        filename = os.path.basename(row["file_path"])
        if "008" in filename:
            # Let's see some samples
            print(f"Row {i}: source={row['dataset_source']}, split={row['split']}, path={row['file_path']}")
