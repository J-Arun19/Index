import os
import torch

class DatasetDecodeError(Exception):
    """Base exception for recoverable sample-level dataset decode/load failures."""
    def __init__(self, message: str, sample_path: str = "", frame_index: int = -1):
        super().__init__(message)
        self.sample_path = sample_path
        self.frame_index = frame_index

class VideoDecodeError(DatasetDecodeError):
    """Exception raised when video opening, seeking, or frame reading fails."""
    pass

class ImageDecodeError(DatasetDecodeError):
    """Exception raised when image file reading or decoding fails."""
    pass

class SamplePreprocessingError(DatasetDecodeError):
    """Exception raised when preprocessing on decoded video frame or image fails."""
    pass

def is_fatal_error(e: BaseException) -> bool:
    """Returns True if the exception is fatal and MUST terminate training immediately."""
    # 1. System interrupts and memory exhaustion
    if isinstance(e, (KeyboardInterrupt, SystemExit, MemoryError)):
        return True
    
    # 2. CUDA / Accelerator errors (CUDA errors must remain fatal)
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    err_type_name = type(e).__name__
    err_str = str(e).lower()
    if "acceleratorerror" in err_type_name.lower() or "acceleratorerror" in err_str:
        return True
    if "cuda" in err_type_name.lower() or "cuda" in err_str:
        return True
    if "out of memory" in err_str:
        return True
        
    # 3. Programming and logic errors
    if isinstance(e, (TypeError, NameError, AttributeError, SyntaxError, ImportError, IndentationError)):
        return True
        
    return False
