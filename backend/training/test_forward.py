import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
from backend.models.image_model import ImageModel
from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn
from torch.utils.data import DataLoader

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    ckpt_path = r"backend/models/weights/runs/full_image_I1/best_checkpoint.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # 1. Initialize model
    model = ImageModel(num_classes=2, pretrained=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Configure layers 1-3 eval mode as in train_image_model
    model.train()
    for name, module in model.backbone.named_modules():
        if ("layer4" not in name and "fc" not in name) and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            
    print("Model loaded and configured.")
    
    # Check all model parameters and buffers in the model
    print("\n--- Model Parameters and Buffers Inspection ---")
    subnormal_buffers = []
    for name, buf in model.named_buffers():
        if torch.isnan(buf).any():
            print(f"BUFFER NaN: {name}")
        if torch.isinf(buf).any():
            print(f"BUFFER Inf: {name}")
        if "running_var" in name:
            min_v = buf.min().item()
            max_v = buf.max().item()
            if min_v < 1e-7:
                subnormal_buffers.append((name, min_v, max_v))
    if subnormal_buffers:
        print(f"Warning: Found {len(subnormal_buffers)} running_var buffers with very small values (< 1e-7):")
        for b_name, b_min, b_max in subnormal_buffers[:10]:
            print(f"  {b_name}: min={b_min:.4e}, max={b_max:.4e}")
    else:
        print("All running_var buffers >= 1e-7.")

    # 2. Test forward pass with synthetic random tensor
    print("\n--- Synthetic Batch Test (Shape: 64, 3, 224, 224) ---")
    dummy_input = torch.randn(64, 3, 224, 224, device=device)
    
    # FP32 forward
    with torch.no_grad():
        out_fp32 = model(dummy_input)
    print(f"FP32 Forward Output: isfinite={torch.isfinite(out_fp32).all().item()}, shape={out_fp32.shape}")
    print(f"  Sample out: {out_fp32[:2]}")
    
    # AMP forward
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            out_amp = model(dummy_input)
    print(f"AMP Forward Output:  isfinite={torch.isfinite(out_amp).all().item()}, shape={out_amp.shape}")
    print(f"  Sample out: {out_amp[:2]}")
    
    # 3. Test with REAL dataset first batch
    print("\n--- Real Dataset First Batch Test ---")
    manifest_path = "backend/training/dataset_manifest.csv"
    train_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=False, 
        collate_fn=safe_collate_fn
    )
    
    real_images, real_labels = next(iter(train_loader))
    print(f"Real Batch Images: shape={real_images.shape}, dtype={real_images.dtype}")
    print(f"  min={real_images.min().item():.4f}, max={real_images.max().item():.4f}, mean={real_images.mean().item():.4f}, std={real_images.std().item():.4f}")
    print(f"  isfinite: {torch.isfinite(real_images).all().item()}")
    print(f"Real Batch Labels: shape={real_labels.shape}, unique={torch.unique(real_labels).tolist()}")
    
    real_images = real_images.to(device)
    real_labels = real_labels.to(device)
    
    # Hook every submodule to see where NaNs first appear
    hook_handles = []
    def create_hook(mod_name):
        def forward_hook(module, input_val, output_val):
            out_tensor = output_val
            if isinstance(out_tensor, tuple):
                out_tensor = out_tensor[0]
            if torch.is_tensor(out_tensor):
                if not torch.isfinite(out_tensor).all():
                    nan_c = torch.isnan(out_tensor).sum().item()
                    inf_c = torch.isinf(out_tensor).sum().item()
                    print(f"  [HOOK ALERT] Layer '{mod_name}' ({module.__class__.__name__}) produced non-finite output! NaNs: {nan_c}, Infs: {inf_c} out of {out_tensor.numel()}")
        return forward_hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0: # leaf modules
            hook_handles.append(module.register_forward_hook(create_hook(name)))
            
    print("\nRunning FP32 forward on real batch...")
    out_real_fp32 = model(real_images)
    print(f"Real Batch FP32 Output: isfinite={torch.isfinite(out_real_fp32).all().item()}")
    print(f"  Sample out: {out_real_fp32[:4]}")
    
    print("\nRunning AMP forward on real batch...")
    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        out_real_amp = model(real_images)
    print(f"Real Batch AMP Output: isfinite={torch.isfinite(out_real_amp).all().item()}")
    print(f"  Sample out: {out_real_amp[:4]}")
    
    # Clean up hooks
    for h in hook_handles:
        h.remove()
        
    # Compare with fresh pretrained model
    print("\n--- Comparing with Fresh Pretrained ResNet18 ---")
    fresh_model = ImageModel(num_classes=2, pretrained=True).to(device)
    fresh_model.train()
    with torch.no_grad():
        fresh_fp32 = fresh_model(real_images)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            fresh_amp = fresh_model(real_images)
    print(f"Fresh Model FP32 Output: isfinite={torch.isfinite(fresh_fp32).all().item()}")
    print(f"Fresh Model AMP Output:  isfinite={torch.isfinite(fresh_amp).all().item()}")

if __name__ == "__main__":
    test()
