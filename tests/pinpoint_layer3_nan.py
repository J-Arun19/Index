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
    collated_imgs, _ = safe_collate_fn(samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageModel(num_classes=2, pretrained=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bb = model.backbone
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=True):
            x = bb.conv1(collated_imgs.to(device))
            x = bb.bn1(x)
            x = bb.relu(x)
            x = bb.maxpool(x)
            x = bb.layer1(x)
            x = bb.layer2(x)
            print("Output of layer2:", x.shape, "min:", x.min().item(), "max:", x.max().item(), "NaN:", torch.isnan(x).sum().item())
            
            # Now step through layer3[0]
            block0 = bb.layer3[0]
            identity = x
            
            out = block0.conv1(x)
            print("block0.conv1:", out.shape, "min:", out.min().item(), "max:", out.max().item(), "NaN:", torch.isnan(out).sum().item(), "dtype:", out.dtype)
            out = block0.bn1(out)
            print("block0.bn1:", out.shape, "min:", out.min().item(), "max:", out.max().item(), "NaN:", torch.isnan(out).sum().item(), "dtype:", out.dtype)
            out = block0.relu(out)
            print("block0.relu:", out.shape, "min:", out.min().item(), "max:", out.max().item(), "NaN:", torch.isnan(out).sum().item(), "dtype:", out.dtype)
            out = block0.conv2(out)
            print("block0.conv2:", out.shape, "min:", out.min().item(), "max:", out.max().item(), "NaN:", torch.isnan(out).sum().item(), "dtype:", out.dtype)
            
            print("Testing block0.bn2 parameters:")
            print("  bn2.running_mean:", block0.bn2.running_mean.min().item(), block0.bn2.running_mean.max().item(), "NaN:", torch.isnan(block0.bn2.running_mean).sum().item())
            print("  bn2.running_var:", block0.bn2.running_var.min().item(), block0.bn2.running_var.max().item(), "NaN:", torch.isnan(block0.bn2.running_var).sum().item())
            print("  bn2.weight:", block0.bn2.weight.min().item(), block0.bn2.weight.max().item(), "NaN:", torch.isnan(block0.bn2.weight).sum().item())
            print("  bn2.bias:", block0.bn2.bias.min().item(), block0.bn2.bias.max().item(), "NaN:", torch.isnan(block0.bn2.bias).sum().item())
            
            out_bn = block0.bn2(out)
            print("block0.bn2 output:", out_bn.shape, "min:", out_bn.min().item(), "max:", out_bn.max().item(), "NaN:", torch.isnan(out_bn).sum().item(), "dtype:", out_bn.dtype)
            
            if block0.downsample is not None:
                print("Testing downsample:")
                ds_conv = block0.downsample[0](identity)
                print("  downsample conv:", ds_conv.shape, "min:", ds_conv.min().item(), "max:", ds_conv.max().item(), "NaN:", torch.isnan(ds_conv).sum().item())
                ds_bn = block0.downsample[1](ds_conv)
                print("  downsample bn:", ds_bn.shape, "min:", ds_bn.min().item(), "max:", ds_bn.max().item(), "NaN:", torch.isnan(ds_bn).sum().item())

if __name__ == "__main__":
    main()
