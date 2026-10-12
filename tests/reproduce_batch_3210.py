import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.utils.config import PipelineConfig
from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn
from backend.training.train_image_model import restore_rng_states
from backend.models.image_model import ImageModel

def main():
    print("=" * 70)
    print("STEP 1: Load Best Checkpoint (Epoch 8) & RNG States")
    print("=" * 70)
    best_path = "backend/models/weights/runs/full_image_I1/best_checkpoint.pth"
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint epoch: {ckpt.get('epoch')}")
    print(f"Config: {ckpt.get('config')}")
    
    # Restore RNG states exactly as train_image_model does
    restore_rng_states(ckpt["random_states"])
    print("RNG states successfully restored.")

    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    print(f"Loading dataset from: {manifest_path}")
    train_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path, use_cache=False)
    print(f"Total dataset samples: {len(train_dataset)}")
    
    batch_size = 16
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "shuffle": True,
        "collate_fn": safe_collate_fn
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)
    print(f"Total batches in train_loader: {len(train_loader)}")

    # We want to identify the exact samples in Batch 3210 (the 3210th batch, 1-indexed)
    # Let's inspect the sampler indices
    # DataLoader with shuffle=True uses RandomSampler
    sampler = train_loader.sampler
    indices = list(sampler)
    print(f"Total sampler indices: {len(indices)}")
    
    batch_3210_indices = indices[3209 * 16 : 3210 * 16]
    print(f"Batch 3210 size: {len(batch_3210_indices)} samples")
    print(f"Batch 3210 dataset indices: {batch_3210_indices}")
    
    print("\n" + "=" * 70)
    print("STEP 2: Inspect individual samples in Batch 3210")
    print("=" * 70)
    
    sample_tensors = []
    sample_labels = []
    
    for i, idx in enumerate(batch_3210_indices):
        sample_meta = train_dataset.samples[idx]
        file_path = sample_meta["file_path"]
        label = sample_meta["label"]
        
        # Load sample through __getitem__
        try:
            img_tensor, lbl_tensor = train_dataset[idx]
        except Exception as e:
            print(f"  Sample {i+1} [idx={idx}]: ERROR loading: {e}")
            continue
            
        if img_tensor is None:
            print(f"  Sample {i+1} [idx={idx}]: img_tensor is None! Path: {file_path}")
            continue
            
        is_finite = torch.isfinite(img_tensor).all().item()
        nan_cnt = torch.isnan(img_tensor).sum().item()
        inf_cnt = torch.isinf(img_tensor).sum().item()
        t_min = img_tensor.min().item()
        t_max = img_tensor.max().item()
        t_mean = img_tensor.mean().item()
        t_std = img_tensor.std().item()
        
        print(f"Sample {i+1:2d} | Idx: {idx:5d} | Label: {label} | Shape: {list(img_tensor.shape)} | Dtype: {img_tensor.dtype} | Min: {t_min:8.4f} | Max: {t_max:8.4f} | Mean: {t_mean:8.4f} | Std: {t_std:8.4f} | NaN: {nan_cnt} | Inf: {inf_cnt} | Finite: {'YES' if is_finite else 'NO'} | Path: {file_path}")
        sample_tensors.append(img_tensor)
        sample_labels.append(lbl_tensor)
        
    print("\n" + "=" * 70)
    print("STEP 3: Collated Batch 3210 Input Tensor Verification")
    print("=" * 70)
    collated_imgs, collated_lbls = safe_collate_fn(list(zip(sample_tensors, sample_labels)))
    print(f"Collated images shape: {collated_imgs.shape}")
    print(f"Collated labels shape: {collated_lbls.shape}")
    
    all_inputs_finite = torch.isfinite(collated_imgs).all().item()
    total_nan = torch.isnan(collated_imgs).sum().item()
    total_inf = torch.isinf(collated_imgs).sum().item()
    b_min = collated_imgs.min().item()
    b_max = collated_imgs.max().item()
    b_mean = collated_imgs.mean().item()
    b_std = collated_imgs.std().item()
    
    print(f"Batch NaN count: {total_nan}")
    print(f"Batch Inf count: {total_inf}")
    print(f"Batch min: {b_min:.6f}")
    print(f"Batch max: {b_max:.6f}")
    print(f"Batch mean: {b_mean:.6f}")
    print(f"Batch std: {b_std:.6f}")
    if all_inputs_finite:
        print("INPUT TENSOR IS FINITE.")
    else:
        print("INPUT TENSOR IS NOT FINITE!")

    print("\n" + "=" * 70)
    print("STEP 4: Forward Pass of Batch 3210 through Epoch 8 Checkpoint Model")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ImageModel(num_classes=2, pretrained=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out_eval = model(collated_imgs.to(device))
    print(f"In model.eval(): Logits shape: {out_eval.shape}, NaN count: {torch.isnan(out_eval).sum().item()}, Inf count: {torch.isinf(out_eval).sum().item()}")
    print(f"Logits min: {out_eval.min().item():.4f}, max: {out_eval.max().item():.4f}")
    print(f"Logits: {out_eval}")
    
    model.train()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out_train = model(collated_imgs.to(device))
    print(f"In model.train(): Logits shape: {out_train.shape}, NaN count: {torch.isnan(out_train).sum().item()}, Inf count: {torch.isinf(out_train).sum().item()}")
    print(f"Logits min: {out_train.min().item():.4f}, max: {out_train.max().item():.4f}")

if __name__ == "__main__":
    main()
