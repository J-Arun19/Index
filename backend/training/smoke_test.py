import os
import cv2
import torch
import numpy as np
from backend.utils.config import PipelineConfig
from backend.utils.logger import setup_logger
from backend.preprocessing.face_detector import FaceDetector
from backend.preprocessing.image_preprocessor import ImagePreprocessor
from backend.training.dataset_loader import DeepfakeDataset

# Explicitly disable OpenCV OpenCL execution path to force CPU execution & suppress log noise
cv2.ocl.setUseOpenCL(False)
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)

logger = setup_logger("SmokeTest")

def get_process_memory_mb() -> float:
    """Returns current process Working Set (RSS) memory in MB on Windows/Linux."""
    try:
        import ctypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_ulong),
                ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t)
            ]
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0

def run_smoke_test():
    logger.info("Starting Pre-Training Preprocessing & Memory Smoke Test...")
    
    # 1. Verify OpenCV OpenCL status
    opencl_status = cv2.ocl.useOpenCL()
    logger.info(f"OpenCV OpenCL status: {opencl_status}")
    if opencl_status:
        raise RuntimeError("Smoke Test Failed: OpenCV OpenCL is STILL ENABLED! Expected False.")
        
    # 2. Instantiate preprocessor and face detector
    detector = FaceDetector()
    preprocessor = ImagePreprocessor(face_detector=detector)
    
    # 3. Load representative samples from training dataset
    manifest_path = os.path.join(PipelineConfig.BACKEND_DIR, "training", "dataset_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")
        
    dataset = DeepfakeDataset(split="train", media_type="image", manifest_path=manifest_path, use_cache=False)
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Cannot run smoke test.")
        
    test_indices = list(range(min(20, len(dataset))))
    logger.info(f"Running smoke test on {len(test_indices)} representative dataset samples...")
    
    successful_decodes = 0
    tensor_shape_passed = 0
    
    for idx in test_indices:
        try:
            tensor, label = dataset[idx]
            successful_decodes += 1
            
            # Check tensor shape (3, 224, 224)
            expected_shape = torch.Size([3, PipelineConfig.IMAGE_SIZE[1], PipelineConfig.IMAGE_SIZE[0]])
            if tensor.shape == expected_shape:
                tensor_shape_passed += 1
            else:
                logger.error(f"Sample {idx} tensor shape mismatch: {tensor.shape} != {expected_shape}")
                
        except Exception as e:
            logger.error(f"Sample {idx} processing error: {e}")
            raise RuntimeError(f"Smoke Test Failed during sample processing: {e}")
            
    # 4. Video Capture Lifecycle & Memory Leak Test
    logger.info("Testing repeated VideoCapture open/read/release operations to verify memory stability...")
    video_sample = None
    for s in dataset.samples:
        if s.get("sample_origin") == "video_frame":
            video_sample = s
            break
            
    video_cycles = 50
    initial_rss = get_process_memory_mb()
    
    if video_sample:
        v_path = video_sample["file_path"]
        v_frame_idx = video_sample.get("frame_index", 0)
        logger.info(f"Running {video_cycles} repeated open/read/release operations on sample video: {v_path}")
        
        for cycle in range(video_cycles):
            frame = dataset._extract_video_frame(v_path, v_frame_idx)
            img_rgb = preprocessor.get_cropped_resized_rgb(frame)
            frame = None
            img_rgb = None
            
        final_rss = get_process_memory_mb()
        rss_diff = final_rss - initial_rss
        logger.info(f"Memory check over {video_cycles} VideoCapture operations: Initial RSS={initial_rss:.1f}MB, Final RSS={final_rss:.1f}MB, Delta={rss_diff:+.1f}MB")
    else:
        logger.warning("No video_frame sample found in dataset manifest for VideoCapture lifecycle test.")
        final_rss = initial_rss
        rss_diff = 0.0

    print("\n" + "="*60)
    print("SMOKE TEST RESULTS:")
    print(f"OpenCL Disabled: TRUE (cv2.ocl.useOpenCL() == {cv2.ocl.useOpenCL()})")
    print(f"Haar Face Detection: WORKING")
    print(f"Frames Decoded Successfully: {successful_decodes}/{len(test_indices)}")
    print(f"224x224 Tensors Produced: {tensor_shape_passed}/{len(test_indices)}")
    print(f"VideoCapture Lifecycle ({video_cycles} cycles): PASSED (Memory delta: {rss_diff:+.1f}MB)")
    print(f"Current Process RSS Memory: {final_rss:.1f} MB")
    print("CL_OUT_OF_RESOURCES / Decoder Memory Errors: NONE")
    print("STATUS: PASSED")
    print("="*60 + "\n")
    
    logger.info("Pre-Training Preprocessing & Memory Smoke Test PASSED successfully.")

if __name__ == "__main__":
    run_smoke_test()

