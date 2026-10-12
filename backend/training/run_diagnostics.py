import os
import sys
import torch
import numpy as np

def inspect_checkpoint(ckpt_path):
    print("=" * 60)
    print(f"INSPECTING CHECKPOINT: {ckpt_path}")
    print("=" * 60)
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found: {ckpt_path}")
        return None
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("Keys in checkpoint:", list(ckpt.keys()))
    print("Epoch:", ckpt.get("epoch"))
    print("Val loss:", ckpt.get("val_loss"))
    print("Train loss:", ckpt.get("train_loss"))
    print("Best val metric:", ckpt.get("best_val_metric"))
    if "config" in ckpt:
        print("Saved Config:", ckpt.get("config"))
    
    # Check model state dict
    model_state = ckpt.get("model_state_dict", {})
    nan_params = []
    inf_params = []
    param_stats = []
    for k, v in model_state.items():
        if torch.is_tensor(v):
            numel = v.numel()
            n_nan = torch.isnan(v).sum().item()
            n_inf = torch.isinf(v).sum().item()
            if n_nan > 0:
                nan_params.append((k, n_nan, numel))
            if n_inf > 0:
                inf_params.append((k, n_inf, numel))
            
            # Print stats for first few and last few params or any extreme values
            v_float = v.float()
            min_val = v_float.min().item() if numel > 0 else 0
            max_val = v_float.max().item() if numel > 0 else 0
            mean_val = v_float.mean().item() if numel > 0 else 0
            std_val = v_float.std().item() if numel > 1 else 0
            param_stats.append((k, min_val, max_val, mean_val, std_val))
            
    print(f"\nModel state parameters checked: {len(model_state)}")
    if nan_params:
        print(f"FAILED: Found NaN in model params: {nan_params}")
    else:
        print("PASSED: No NaNs found in model state dict.")
    if inf_params:
        print(f"FAILED: Found Inf in model params: {inf_params}")
    else:
        print("PASSED: No Infs found in model state dict.")
        
    print("\nSample parameter statistics (first 5 and last 5):")
    for stat in param_stats[:5] + param_stats[-5:]:
        print(f"  {stat[0]}: min={stat[1]:.4e}, max={stat[2]:.4e}, mean={stat[3]:.4e}, std={stat[4]:.4e}")

    # Check optimizer state dict
    opt_state = ckpt.get("optimizer_state_dict")
    if opt_state is not None:
        print("\nOptimizer state dict keys:", list(opt_state.keys()))
        opt_nan = []
        opt_inf = []
        state_dict = opt_state.get("state", {})
        print(f"Optimizer tracked parameters: {len(state_dict)}")
        for p_id, p_dict in state_dict.items():
            for s_name, s_val in p_dict.items():
                if torch.is_tensor(s_val):
                    n_nan = torch.isnan(s_val).sum().item()
                    n_inf = torch.isinf(s_val).sum().item()
                    if n_nan > 0:
                        opt_nan.append((p_id, s_name, n_nan))
                    if n_inf > 0:
                        opt_inf.append((p_id, s_name, n_inf))
        if opt_nan:
            print(f"FAILED: Optimizer state contains NaNs: {opt_nan}")
        else:
            print("PASSED: No NaNs in optimizer state.")
        if opt_inf:
            print(f"FAILED: Optimizer state contains Infs: {opt_inf}")
        else:
            print("PASSED: No Infs in optimizer state.")
            
        param_groups = opt_state.get("param_groups", [])
        print(f"Optimizer param groups: {len(param_groups)}")
        for idx, pg in enumerate(param_groups):
            pg_info = {k: v for k, v in pg.items() if k != "params"}
            print(f"  Group {idx}: num_params={len(pg.get('params', []))}, {pg_info}")
    else:
        print("\nNo optimizer state dict in checkpoint.")
        
    # Check scheduler state dict
    sched_state = ckpt.get("scheduler_state_dict")
    print(f"\nScheduler state dict: {sched_state}")
    
    return ckpt

if __name__ == "__main__":
    best_path = r"backend/models/weights/runs/full_image_I1/best_checkpoint.pth"
    latest_path = r"backend/models/weights/runs/full_image_I1/latest_checkpoint.pth"
    inspect_checkpoint(best_path)
    inspect_checkpoint(latest_path)
