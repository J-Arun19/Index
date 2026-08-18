import json
import os

splits_dir = r"d:\Download\satya-vision-guard-main\satya-vision-guard-main\backend\training\splits"

def load_ids(name):
    ids = set()
    with open(os.path.join(splits_dir, f"{name}.json"), "r") as f:
        pairs = json.load(f)
        for pair in pairs:
            for seq in pair:
                ids.add(str(seq).zfill(3))
    return ids

train = load_ids("train")
val = load_ids("val")
test = load_ids("test")

print("Train len:", len(train))
print("Val len:", len(val))
print("Test len:", len(test))

print("Train & Val overlap:", train.intersection(val))
print("Train & Test overlap:", train.intersection(test))
print("Val & Test overlap:", val.intersection(test))
