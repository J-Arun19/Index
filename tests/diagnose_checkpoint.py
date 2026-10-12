import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def inspect_checkpoint(path):
    print("=" * 70)
    print(f"CHECKPOINT INSPECTION: {path}")
    print("=" * 70)
    if not os.path.exists(path):
        print("  File not found!")
        return None
    
    file_size = os.path.getsize(path)
    print(f"  File size: {file_size:,} bytes")
    
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        print("  File integrity: PASS (cleanly unpickled)")
    except Exception as e:
        print(f"  File integrity: FAIL ({e})")
        return None
        
    print(f"  Top-level keys: {list(ckpt.keys())}")
    print(f"  Epoch: {ckpt.get('epoch')}")
    print(f"  Val Loss: {ckpt.get('val_loss')}")
    print(f"  Best Val Metric: {ckpt.get('best_val_metric')}")
    print(f"  Optimal Threshold: {ckpt.get('optimal_threshold')}")
    print(f"  Val Metrics: {ckpt.get('val_metrics')}")
    print(f"  Config: {ckpt.get('config')}")
    
    # Model parameters inspection
    model_sd = ckpt.get("model_state_dict", {})
    param_count = len(model_sd)
    nan_params = []
    inf_params = []
    max_abs_val = 0.0
    total_elements = 0
    
    for name, tensor in model_sd.items():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            total_elements += tensor.numel()
            n_nan = torch.isnan(tensor).sum().item()
            n_inf = torch.isinf(tensor).sum().item()
            if n_nan > 0:
                nan_params.append((name, n_nan))
            if n_inf > 0:
                inf_params.append((name, n_inf))
            cur_max = tensor.abs().max().item()
            if cur_max > max_abs_val:
                max_abs_val = cur_max
                
    model_finite = (len(nan_params) == 0 and len(inf_params) == 0)
    print(f"  Model Parameters Total Tensors: {param_count}")
    print(f"  Model Parameters Total Float Elements: {total_elements:,}")
    print(f"  Model Parameters Finite: {'YES' if model_finite else 'NO'}")
    print(f"  Global Maximum Absolute Parameter Value: {max_abs_val:.8e}")
    if not model_finite:
        print(f"    Layers with NaN: {nan_params}")
        print(f"    Layers with Inf: {inf_params}")
        
    # Optimizer state inspection
    opt_sd = ckpt.get("optimizer_state_dict")
    opt_finite = True
    opt_max_abs = 0.0
    opt_nan_entries = []
    opt_inf_entries = []
    if opt_sd and "state" in opt_sd:
        print(f"  Optimizer Type in state: Adam / param groups count: {len(opt_sd.get('param_groups', []))}")
        for pg_idx, pg in enumerate(opt_sd.get('param_groups', [])):
            print(f"    Param Group {pg_idx}: lr={pg.get('lr')}, weight_decay={pg.get('weight_decay')}")
        for p_id, p_state in opt_sd["state"].items():
            for k, v in p_state.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    n_nan = torch.isnan(v).sum().item()
                    n_inf = torch.isinf(v).sum().item()
                    if n_nan > 0:
                        opt_nan_entries.append((p_id, k, n_nan))
                    if n_inf > 0:
                        opt_inf_entries.append((p_id, k, n_inf))
                    cur_max = v.abs().max().item()
                    if cur_max > opt_max_abs:
                        opt_max_abs = cur_max
        opt_finite = (len(opt_nan_entries) == 0 and len(opt_inf_entries) == 0)
        print(f"  Optimizer State Finite: {'YES' if opt_finite else 'NO'}")
        print(f"  Global Maximum Absolute Optimizer State Value: {opt_max_abs:.8e}")
        if not opt_finite:
            print(f"    Opt states with NaN: {opt_nan_entries}")
            print(f"    Opt states with Inf: {opt_inf_entries}")
    else:
        print("  Optimizer state not found in checkpoint")
        
    # Scheduler state
    sched_sd = ckpt.get("scheduler_state_dict")
    print(f"  Scheduler State: {sched_sd}")
    
    # Scaler state
    scaler_sd = ckpt.get("scaler_state_dict")
    print(f"  Scaler State: {scaler_sd}")
    
    return ckpt

if __name__ == "__main__":
    latest_path = "backend/models/weights/runs/full_image_I1/latest_checkpoint.pth"
    best_path = "backend/models/weights/runs/full_image_I1/best_checkpoint.pth"
    inspect_checkpoint(latest_path)
    print()
    inspect_checkpoint(best_path)
