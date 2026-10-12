import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from backend.models.image_model import ImageModel
from backend.training.train_image_model import (
    NumericalInstabilityError,
    check_tensor_finite,
    check_labels_finite,
    validate_checkpoint_state,
    save_checkpoint_atomic,
)


def test_nan_input_detection():
    print("\n[TEST 1] NaN input detection...")
    bad_images = torch.randn(4, 3, 224, 224)
    bad_images[1, 0, 10, 10] = float('nan')

    caught = False
    try:
        check_tensor_finite(
            bad_images,
            stage="1_input_tensor",
            tensor_name="images",
            epoch=9,
            batch_idx=42,
            lr=1e-4,
            sample_info="Batch [42/3200], size=4"
        )
    except NumericalInstabilityError as e:
        caught = True
        assert e.stage == "1_input_tensor", f"Expected stage 1_input_tensor, got {e.stage}"
        assert e.tensor_name == "images", f"Expected tensor_name images, got {e.tensor_name}"
        assert "NaN Count: 1" in e.details, "Expected NaN Count: 1 in error details"
        print(f"  --> Correctly caught: {e}")

    assert caught, "Failed to catch NaN in input tensor!"
    print("  --> PASS: NaN input detected and halted immediately.")


def test_nan_logits_detection():
    print("\n[TEST 2] NaN logits detection...")
    bad_logits = torch.randn(4, 2)
    bad_logits[0, 1] = float('nan')

    caught = False
    try:
        check_tensor_finite(
            bad_logits,
            stage="3_model_logits",
            tensor_name="outputs",
            epoch=9,
            batch_idx=100,
            lr=1e-4
        )
    except NumericalInstabilityError as e:
        caught = True
        assert e.stage == "3_model_logits"
        assert e.tensor_name == "outputs"
        print(f"  --> Correctly caught: {e}")

    assert caught, "Failed to catch NaN in model logits!"
    print("  --> PASS: NaN logits detected and halted immediately.")


def test_nan_loss_detection():
    print("\n[TEST 3] NaN loss detection...")
    nan_loss = torch.tensor(float('nan'))

    caught = False
    try:
        check_tensor_finite(
            nan_loss,
            stage="4_loss",
            tensor_name="cross_entropy_loss",
            epoch=9,
            batch_idx=250,
            lr=1e-4
        )
    except NumericalInstabilityError as e:
        caught = True
        assert e.stage == "4_loss"
        print(f"  --> Correctly caught: {e}")

    assert caught, "Failed to catch NaN loss!"
    print("  --> PASS: NaN loss detected and halted before backward pass.")


def test_inf_loss_detection():
    print("\n[TEST 4] Inf loss detection (+Inf and -Inf)...")
    pos_inf_loss = torch.tensor(float('inf'))
    neg_inf_loss = torch.tensor(float('-inf'))

    for loss_val, sign in [(pos_inf_loss, "+Inf"), (neg_inf_loss, "-Inf")]:
        caught = False
        try:
            check_tensor_finite(
                loss_val,
                stage="4_loss",
                tensor_name="cross_entropy_loss",
                epoch=9,
                batch_idx=250,
                lr=1e-4
            )
        except NumericalInstabilityError as e:
            caught = True
            assert e.stage == "4_loss"
            assert "Inf Count: 1" in e.details
            print(f"  --> Correctly caught {sign} loss: {e}")
        assert caught, f"Failed to catch {sign} loss!"

    print("  --> PASS: Both +Inf and -Inf loss detected and halted before backward pass.")


def test_nan_gradient_detection():
    print("\n[TEST 5] NaN gradient detection...")
    model = nn.Linear(10, 2)
    x = torch.randn(4, 10)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Inject NaN into weight gradient
    model.weight.grad[0, 0] = float('nan')

    caught = False
    try:
        for p_name, p in model.named_parameters():
            if p.grad is not None:
                check_tensor_finite(
                    p.grad,
                    stage="5_gradients",
                    tensor_name=f"grad_{p_name}",
                    epoch=9,
                    batch_idx=500,
                    lr=1e-4
                )
    except NumericalInstabilityError as e:
        caught = True
        assert e.stage == "5_gradients"
        assert "grad_weight" in e.tensor_name
        print(f"  --> Correctly caught: {e}")

    assert caught, "Failed to catch NaN in gradient!"
    print("  --> PASS: NaN gradient detected and halted before optimizer step.")


def test_inf_gradient_detection():
    print("\n[TEST 6] Inf gradient detection...")
    model = nn.Linear(10, 2)
    x = torch.randn(4, 10)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Inject Inf into bias gradient
    model.bias.grad[0] = float('inf')

    caught = False
    try:
        for p_name, p in model.named_parameters():
            if p.grad is not None:
                check_tensor_finite(
                    p.grad,
                    stage="5_gradients",
                    tensor_name=f"grad_{p_name}",
                    epoch=9,
                    batch_idx=501,
                    lr=1e-4
                )
    except NumericalInstabilityError as e:
        caught = True
        assert e.stage == "5_gradients"
        assert "grad_bias" in e.tensor_name
        print(f"  --> Correctly caught: {e}")

    assert caught, "Failed to catch Inf in gradient!"
    print("  --> PASS: Inf gradient detected and halted before optimizer step.")


def test_corrupted_model_parameter_detection():
    print("\n[TEST 7] Corrupted model parameter/buffer detection...")
    model = ImageModel(num_classes=2, pretrained=False)

    # Simulate BatchNorm running_mean corruption (as occurred in Epoch 9)
    bn_buf = model.backbone.layer4[0].bn1.running_mean
    bn_buf[0] = float('nan')

    caught = False
    try:
        for b_name, buf in model.named_buffers():
            check_tensor_finite(
                buf.data,
                stage="7_model_buffer",
                tensor_name=b_name,
                epoch=9,
                batch_idx=600,
                lr=1e-4
            )
    except NumericalInstabilityError as e:
        caught = True
        assert e.stage == "7_model_buffer"
        assert "backbone.layer4.0.bn1.running_mean" in e.tensor_name
        print(f"  --> Correctly caught buffer corruption: {e}")

    assert caught, "Failed to catch corrupted BatchNorm buffer!"
    print("  --> PASS: Corrupted model parameter/buffer detected post-step.")


def test_corrupted_optimizer_state_detection():
    print("\n[TEST 8] Corrupted optimizer state detection...")
    model = nn.Linear(5, 2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run one step to initialize optimizer state dict
    x = torch.randn(2, 5)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    opt_state = optimizer.state_dict()

    # Inject NaN into exp_avg (momentum buffer)
    first_param_id = list(opt_state["state"].keys())[0]
    opt_state["state"][first_param_id]["exp_avg"][0, 0] = float('nan')

    test_state = {
        "epoch": 9,
        "train_loss": 0.25,
        "val_loss": 0.35,
        "best_val_metric": 0.75,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt_state
    }

    is_valid, errors = validate_checkpoint_state(test_state)
    assert not is_valid, "Failed to identify corrupted optimizer state!"
    assert any("Optimizer param" in err and "exp_avg" in err for err in errors), f"Unexpected errors: {errors}"
    print(f"  --> Correctly flagged optimizer corruption: {errors}")
    print("  --> PASS: Corrupted optimizer state detected.")


def test_corrupted_checkpoint_cannot_overwrite_healthy():
    print("\n[TEST 9] Corrupted checkpoint overwrite prevention...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "latest_checkpoint.pth")
        diag_path = os.path.join(tmp_dir, "latest_checkpoint_corrupted_diagnostic.pth")

        # 1. Save known-good checkpoint
        healthy_state = {
            "epoch": 8,
            "train_loss": 0.1660,
            "val_loss": 0.3864,
            "best_val_metric": 0.7672,
            "model_state_dict": {"fc.weight": torch.tensor([[1.0, 2.0]])},
            "optimizer_state_dict": None
        }
        res = save_checkpoint_atomic(healthy_state, ckpt_path)
        assert res == True, "Failed to save healthy checkpoint"
        assert os.path.exists(ckpt_path), "Healthy checkpoint not found"

        initial_mtime = os.path.getmtime(ckpt_path)
        initial_content = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert initial_content["epoch"] == 8

        # 2. Attempt to save corrupted Epoch 9 checkpoint
        corrupted_state = {
            "epoch": 9,
            "train_loss": float('nan'),
            "val_loss": float('nan'),
            "best_val_metric": 0.1737,
            "model_state_dict": {"fc.weight": torch.tensor([[float('nan'), 2.0]])},
            "optimizer_state_dict": None
        }
        saved = save_checkpoint_atomic(corrupted_state, ckpt_path)
        assert saved == False, "Corrupted checkpoint was erroneously saved!"

        # 3. Verify healthy checkpoint was NOT overwritten
        post_attempt_content = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert post_attempt_content["epoch"] == 8, f"Checkpoint was overwritten! Found epoch: {post_attempt_content['epoch']}"
        assert torch.equal(post_attempt_content["model_state_dict"]["fc.weight"], torch.tensor([[1.0, 2.0]]))

        # 4. Verify diagnostic file was saved separately
        assert os.path.exists(diag_path), f"Diagnostic checkpoint not created at {diag_path}"
        diag_content = torch.load(diag_path, map_location="cpu", weights_only=False)
        assert diag_content["epoch"] == 9
        print("  --> PASS: Healthy checkpoint preserved; corrupted checkpoint refused and saved to diagnostic file.")


def test_healthy_checkpoint_loads_successfully():
    print("\n[TEST 10] Healthy checkpoint loading validation...")
    best_path = "backend/models/weights/runs/full_image_I1/best_checkpoint.pth"
    assert os.path.exists(best_path), f"best_checkpoint.pth not found at {best_path}"

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    is_valid, errors = validate_checkpoint_state(ckpt)
    assert is_valid == True, f"best_checkpoint.pth failed validation: {errors}"
    assert errors == []

    assert ckpt.get("epoch") in (8, 9), f"Expected Epoch 8 or 9, got {ckpt.get('epoch')}"
    assert np.isfinite(ckpt.get("train_loss")), "Train loss non-finite in best checkpoint"
    assert np.isfinite(ckpt.get("val_loss")), "Val loss non-finite in best checkpoint"
    assert np.isfinite(ckpt.get("best_val_metric")), "Best val metric non-finite in best checkpoint"

    # Verify ImageModel can load the weights cleanly
    model = ImageModel(num_classes=2, pretrained=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Verify model forward pass on dummy input produces finite output
    test_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(test_input)
    assert torch.isfinite(out).all(), "Forward pass on best_checkpoint produced non-finite output!"
    print(f"  --> Successfully loaded Epoch {ckpt.get('epoch')} best checkpoint: Val Macro-F1 = {ckpt.get('best_val_metric'):.4f}")
    print("  --> PASS: Healthy checkpoint loaded, validated, and verified.")


def run_all_tests():
    print("=" * 70)
    print("SATYADRISHTI NUMERICAL STABILITY REGRESSION TEST SUITE")
    print("=" * 70)

    test_nan_input_detection()
    test_nan_logits_detection()
    test_nan_loss_detection()
    test_inf_loss_detection()
    test_nan_gradient_detection()
    test_inf_gradient_detection()
    test_corrupted_model_parameter_detection()
    test_corrupted_optimizer_state_detection()
    test_corrupted_checkpoint_cannot_overwrite_healthy()
    test_healthy_checkpoint_loads_successfully()

    print("\n" + "=" * 70)
    print("ALL 10 NUMERICAL STABILITY REGRESSION TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
