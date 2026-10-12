import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.optim as optim
from backend.models.image_model import ImageModel
from backend.training.dataset_loader import DeepfakeDataset, safe_collate_fn
from torch.utils.data import DataLoader
from backend.utils.config import PipelineConfig

def test_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    
    ckpt_path = r"backend/models/weights/runs/full_image_I1/best_checkpoint.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    model = ImageModel(num_classes=2, pretrained=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Configure layers 1-3 eval mode as in train_image_model
    model.train()
    for name, module in model.backbone.named_modules():
        if ("layer4" not in name and "fc" not in name) and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            
    # Test cuDNN settings with AMP
    print("\n--- Testing cuDNN vs AMP ---")
    dummy = torch.randn(64, 3, 224, 224, device=device)
    
    print("Test 1: AMP with default cuDNN:")
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            out1 = model(dummy)
    print("  Output isfinite:", torch.isfinite(out1).all().item())
    
    print("Test 2: AMP with torch.backends.cudnn.enabled = False:")
    torch.backends.cudnn.enabled = False
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            out2 = model(dummy)
    print("  Output isfinite:", torch.isfinite(out2).all().item())
    torch.backends.cudnn.enabled = True
    
    # Test FP32 Complete Forward + Backward + Optimizer Step
    print("\n--- Testing Full FP32 Diagnostic Step (No Weight Updates) ---")
    manifest_path = "backend/training/dataset_manifest.csv"
    train_dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=False, 
        collate_fn=safe_collate_fn
    )
    
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    # FP32 forward pass
    outputs = model(images)
    print("Outputs shape:", outputs.shape, "isfinite:", torch.isfinite(outputs).all().item())
    print("Outputs min/max:", outputs.min().item(), outputs.max().item())
    
    loss = criterion(outputs, labels)
    print("Loss:", loss.item(), "isfinite:", torch.isfinite(loss).item())
    
    loss.backward()
    
    nan_grads = []
    max_grad = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                nan_grads.append(name)
            else:
                max_grad = max(max_grad, p.grad.abs().max().item())
                
    print(f"Gradients computed: {len(nan_grads)} non-finite params.")
    if nan_grads:
        print("  Non-finite gradient params:", nan_grads)
    else:
        print(f"  All gradients are finite! Max abs grad: {max_grad:.4e}")
        
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"Gradient norm: {grad_norm.item():.4f}, isfinite: {torch.isfinite(grad_norm).item()}")

if __name__ == "__main__":
    test_step()
