import subprocess
import os
import sys
import json
import glob
import time

# Set pythonpath
env = os.environ.copy()
workspace_dir = r"d:\Download\satya-vision-guard-main\satya-vision-guard-main"
env["PYTHONPATH"] = workspace_dir

python_exe = os.path.join(workspace_dir, ".venv", "Scripts", "python")

# We will run both image and video models across the matrix
image_experiments = [
    # 1. Baseline (already run or will run)
    {"name": "Baseline", "args": ["--pilot", "--strategy", "standard", "--no-augmentation", "--epochs", "2", "--batch-size", "128"]},
    # 2. Augmented
    {"name": "Augmented", "args": ["--pilot", "--strategy", "standard", "--epochs", "2", "--batch-size", "128"]},
    # 3. Pretrained/Tuned
    {"name": "Pretrained", "args": ["--pilot", "--strategy", "standard", "--pretrained", "--epochs", "2", "--batch-size", "128"]},
    # 4. Weighted Loss
    {"name": "Weighted Loss", "args": ["--pilot", "--strategy", "weighted_loss", "--pretrained", "--epochs", "2", "--batch-size", "128"]},
    # 5. Sampler
    {"name": "Sampler", "args": ["--pilot", "--strategy", "sampler", "--pretrained", "--epochs", "2", "--batch-size", "128"]},
    # 6. Both
    {"name": "Both", "args": ["--pilot", "--strategy", "both", "--pretrained", "--epochs", "2", "--batch-size", "128"]},
]

video_experiments = [
    # 1. Baseline
    {"name": "Baseline", "args": ["--pilot", "--strategy", "standard", "--no-augmentation", "--epochs", "2", "--batch-size", "128"]},
    # 2. Augmented
    {"name": "Augmented", "args": ["--pilot", "--strategy", "standard", "--epochs", "2", "--batch-size", "128"]},
    # 3. Pretrained/Tuned
    {"name": "Pretrained", "args": ["--pilot", "--strategy", "standard", "--pretrained", "--epochs", "2", "--batch-size", "128"]},
    # 4. Weighted Loss
    {"name": "Weighted Loss", "args": ["--pilot", "--strategy", "weighted_loss", "--pretrained", "--epochs", "2", "--batch-size", "128"]},
    # 5. Sampler
    {"name": "Sampler", "args": ["--pilot", "--strategy", "sampler", "--pretrained", "--epochs", "2", "--batch-size", "128"]},
    # 6. Both
    {"name": "Both", "args": ["--pilot", "--strategy", "both", "--pretrained", "--epochs", "2", "--batch-size", "128"]},
]

def run_cmd(script_name, name, args):
    cmd = [python_exe, os.path.join(workspace_dir, "backend", "training", script_name)] + args
    print(f"\n========================================\nRunning: {script_name} ({name})\nArguments: {' '.join(args)}\n========================================\n")
    start_time = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    duration = time.time() - start_time
    print(f"Completed in {duration:.2f} seconds.")
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        # We don't want to crash the whole matrix if one config fails, just continue
    else:
        print(f"Successfully finished: {script_name} ({name})")
        # Print last few lines of stdout
        lines = result.stdout.splitlines()
        for line in lines[-15:]:
            print(line)

def main():
    print("Starting controlled pilot experiment matrix...")
    
    # Check if a model type is specified as argument (image or video)
    model_type = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    
    if model_type in ("image", "all"):
        print("\n========================================\n=== Running Image Model Experiments ===\n========================================\n")
        # If the first baseline run is already done, we can skip the first one or rerun it
        # Let's check if baseline_metrics.json exists in image results dir
        image_results_dir = os.path.join(workspace_dir, "backend", "training", "results", "image")
        baseline_exists = os.path.exists(os.path.join(image_results_dir, "baseline_metrics.json"))
        
        for exp in image_experiments:
            if exp["name"] == "Baseline" and baseline_exists:
                print("Image Baseline run already exists. Skipping baseline run to save time.")
                continue
            run_cmd("train_image_model.py", exp["name"], exp["args"])
            # Mark baseline_exists true if we just ran it
            if exp["name"] == "Baseline":
                baseline_exists = True
        
    if model_type in ("video", "all"):
        print("\n========================================\n=== Running Video Model Experiments ===\n========================================\n")
        video_results_dir = os.path.join(workspace_dir, "backend", "training", "results", "video")
        baseline_exists = os.path.exists(os.path.join(video_results_dir, "baseline_metrics.json"))
        
        for exp in video_experiments:
            if exp["name"] == "Baseline" and baseline_exists:
                print("Video Baseline run already exists. Skipping baseline run to save time.")
                continue
            run_cmd("train_video_model.py", exp["name"], exp["args"])
            # Mark baseline_exists true if we just ran it
            if exp["name"] == "Baseline":
                baseline_exists = True

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
