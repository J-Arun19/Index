import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.utils.config import PipelineConfig
from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn
from backend.training.train_image_model import restore_rng_states
from backend.models.image_model import ImageModel

def main():
    best_path = "backend/models/weights/runs/full_image_I1/best_checkpoint.pth"
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    restore_rng_states(ckpt["random_states"])

    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    train_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path, use_cache=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=safe_collate_fn)
    indices = list(train_loader.sampler)
    batch_3210_indices = indices[3209 * 16 : 3210 * 16]
    
    samples = [train_dataset[i] for i in batch_3210_indices]
    collated_imgs, collated_lbls = safe_collate_fn(samples)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageModel(num_classes=2, pretrained=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print("=" * 70)
    print("TEST A: Forward Pass in FP32 on CUDA (NO AMP)")
    print("=" * 70)
    with torch.no_grad():
        out_fp32 = model(collated_imgs.to(device))
    print(f"FP32 CUDA - Logits NaN count: {torch.isnan(out_fp32).sum().item()}, Inf count: {torch.isinf(out_fp32).sum().item()}")
    print(f"FP32 CUDA - Logits: {out_fp32}")
    
    print("\n" + "=" * 70)
    print("TEST B: Forward Pass in FP32 on CPU")
    print("=" * 70)
    model_cpu = ImageModel(num_classes=2, pretrained=True)
    model_cpu.load_state_dict(ckpt["model_state_dict"])
    model_cpu.eval()
    with torch.no_grad():
        out_cpu = model_cpu(collated_imgs.cpu())
    print(f"FP32 CPU - Logits NaN count: {torch.isnan(out_cpu).sum().item()}, Inf count: {torch.isinf(out_cpu).sum().item()}")
    print(f"FP32 CPU - Logits: {out_cpu}")
    
    print("\n" + "=" * 70)
    print("TEST C: Layer-by-Layer Forward Pass with AMP (FP16)")
    print("=" * 70)
    x = collated_imgs.to(device)
    bb = model.backbone
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            print(f"Input: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = bb.conv1(x)
            print(f"conv1: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = bb.bn1(x)
            print(f"bn1: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = bb.relu(x)
            print(f"relu: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = bb.maxpool(x)
            print(f"maxpool: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            
            x = bb.layer1(x)
            print(f"layer1: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = bb.layer2(x)
            print(f"layer2: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = bb.layer3(x)
            print(f"layer3: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = bb.layer4(x)
            print(f"layer4: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = bb.avgpool(x)
            print(f"avgpool: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = torch.flatten(x, 1)
            print(f"flatten: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = bb.fc(x)
            print(f"fc: shape={x.shape}, NaN={torch.isnan(x).sum().item()}, min={x.min().item():.4f}, max={x.max().item():.4f}")

    print("\n" + "=" * 70)
    print("TEST D: Check individual samples in Batch 3210 through model with AMP")
    print("=" * 70)
    for s_idx in range(len(samples)):
        s_img, s_lbl = samples[s_idx]
        single_batch = s_img.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                single_out = model(single_batch)
        s_nan = torch.isnan(single_out).sum().item()
        print(f"Sample {s_idx+1:2d} (dataset idx {batch_3210_indices[s_idx]}): NaN count: {s_nan}, logits: {single_out}")

if __name__ == "__main__":
    main()
