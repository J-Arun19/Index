import os
import threading
import torch
from typing import Optional, Tuple, Dict, Any, Callable, TypeVar
from backend.utils.logger import setup_logger

logger = setup_logger("DeviceManager")

T = TypeVar("T")


class DeviceManager:
    """
    Centralized, thread-safe inference device manager.
    Validates CUDA health via active runtime probes before dispatching work,
    tracks CUDA context health, and orchestrates safe transitions to CPU fallback.
    """

    _lock = threading.Lock()
    _cuda_healthy: Optional[bool] = None
    _cuda_unhealthy_reason: Optional[str] = None
    _device_override: Optional[str] = None

    @classmethod
    def probe_cuda(cls, force: bool = False) -> Tuple[bool, str]:
        """
        Actively verifies whether the CUDA runtime and driver context are operational.
        Does not rely solely on torch.cuda.is_available(). Allocates a small probe tensor
        and performs a minimal calculation inside a guarded try/except block.
        """
        with cls._lock:
            if not force and cls._cuda_healthy is not None:
                return cls._cuda_healthy, cls._cuda_unhealthy_reason or "Cached status"

            if not torch.cuda.is_available():
                cls._cuda_healthy = False
                cls._cuda_unhealthy_reason = "torch.cuda.is_available() is False"
                logger.info("DeviceManager: CUDA unavailable on system; default is CPU.")
                return False, cls._cuda_unhealthy_reason

            try:
                # Active probe: allocate and execute a tiny operation on GPU
                probe_tensor = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
                _ = (probe_tensor + 1.0).sum().item()
                del probe_tensor

                cls._cuda_healthy = True
                cls._cuda_unhealthy_reason = None
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA:0"
                logger.info(f"DeviceManager: CUDA runtime probe succeeded on {gpu_name}.")
                return True, "CUDA runtime probe succeeded"
            except Exception as e:
                cls._cuda_healthy = False
                cls._cuda_unhealthy_reason = f"{type(e).__name__}: {str(e)}"
                logger.warning(
                    f"DeviceManager: CUDA runtime probe failed ({cls._cuda_unhealthy_reason}). "
                    f"Marking CUDA as unhealthy and selecting CPU."
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return False, cls._cuda_unhealthy_reason

    @classmethod
    def is_cuda_healthy(cls) -> bool:
        """Returns True only if CUDA is available and has passed active runtime validation."""
        if cls._cuda_healthy is None:
            healthy, _ = cls.probe_cuda()
            return healthy
        return cls._cuda_healthy

    @classmethod
    def mark_cuda_unhealthy(cls, reason: str) -> None:
        """
        Marks CUDA context as compromised. All subsequent device requests
        will cleanly route to CPU to avoid repeatedly hitting a corrupted CUDA driver state.
        """
        with cls._lock:
            cls._cuda_healthy = False
            cls._cuda_unhealthy_reason = reason
            logger.error(
                f"DeviceManager: CUDA marked unhealthy. Reason: {reason}. "
                f"All future inference will route to CPU."
            )
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    @classmethod
    def get_device(cls, preferred: Optional[str] = None) -> torch.device:
        """
        Returns a validated torch.device instance.
        If preferred is 'cpu', returns cpu.
        If preferred is 'cuda' (or None), returns cuda only if active probe succeeds,
        otherwise falls back to cpu.
        """
        if cls._device_override:
            return torch.device(cls._device_override)

        if preferred == "cpu":
            return torch.device("cpu")

        if cls.is_cuda_healthy():
            return torch.device("cuda")

        return torch.device("cpu")

    @classmethod
    def get_device_str(cls, preferred: Optional[str] = None) -> str:
        """Returns device name as string ('cuda' or 'cpu')."""
        return cls.get_device(preferred).type

    @classmethod
    def safe_move_model(cls, model: torch.nn.Module, target_device: torch.device) -> Tuple[torch.nn.Module, torch.device]:
        """
        Safely moves model to target_device. If moving to CUDA fails,
        catches the error, marks CUDA unhealthy, and places model on CPU.
        """
        if target_device.type == "cuda":
            if not cls.is_cuda_healthy():
                logger.info("DeviceManager: CUDA is unhealthy; keeping model on CPU.")
                model.to("cpu")
                return model, torch.device("cpu")
            try:
                model.to(target_device)
                return model, target_device
            except Exception as e:
                cls.mark_cuda_unhealthy(f"Failed moving model to {target_device}: {e}")
                model.to("cpu")
                return model, torch.device("cpu")
        else:
            model.to("cpu")
            return model, torch.device("cpu")

    @classmethod
    def safe_inference_call(
        cls,
        model_name: str,
        stage: str,
        primary_fn: Callable[[], T],
        cpu_fallback_fn: Optional[Callable[[], T]] = None,
    ) -> T:
        """
        Executes primary inference function. If a CUDA-related error occurs,
        marks CUDA unhealthy, cleans up GPU tensor memory, and retries using cpu_fallback_fn.
        """
        try:
            return primary_fn()
        except Exception as e:
            err_str = str(e)
            is_cuda_err = (
                "cuda" in err_str.lower()
                or "out of memory" in err_str.lower()
                or isinstance(e, torch.cuda.CudaError)
            )

            logger.warning(
                f"DeviceManager: {model_name} failed at stage '{stage}' "
                f"(is_cuda_error={is_cuda_err}): {type(e).__name__}: {err_str}"
            )

            if is_cuda_err:
                cls.mark_cuda_unhealthy(f"{model_name} failed at stage '{stage}': {err_str}")

            if cpu_fallback_fn is not None and is_cuda_err:
                logger.info(f"DeviceManager: Initiating safe CPU fallback for {model_name} (stage: {stage})...")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                try:
                    result = cpu_fallback_fn()
                    logger.info(f"DeviceManager: CPU fallback for {model_name} succeeded.")
                    return result
                except Exception as cpu_err:
                    logger.error(f"DeviceManager: CPU fallback for {model_name} also failed: {cpu_err}")
                    raise RuntimeError(
                        f"Specialist {model_name} failed on GPU ({err_str}) "
                        f"and CPU fallback also failed: {str(cpu_err)}"
                    ) from cpu_err

            raise

    @classmethod
    def get_cuda_status(cls) -> Dict[str, Any]:
        """Provides full inspection details of the host PyTorch and CUDA runtime."""
        status: Dict[str, Any] = {
            "pytorch_version": torch.__version__,
            "torch_built_with_cuda": torch.version.cuda is not None,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_healthy": cls.is_cuda_healthy(),
            "unhealthy_reason": cls._cuda_unhealthy_reason,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "selected_device": cls.get_device_str(),
        }

        if torch.cuda.is_available() and cls.is_cuda_healthy() and torch.cuda.device_count() > 0:
            try:
                status["gpu_name"] = torch.cuda.get_device_name(0)
                cap = torch.cuda.get_device_capability(0)
                status["compute_capability"] = f"{cap[0]}.{cap[1]}"
                props = torch.cuda.get_device_properties(0)
                status["total_vram_mb"] = round(props.total_memory / (1024 ** 2), 1)
                status["allocated_vram_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
                status["reserved_vram_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)
            except Exception as e:
                status["gpu_inspect_error"] = str(e)

        return status

    @classmethod
    def set_device_override(cls, device_str: Optional[str]) -> None:
        """Sets an explicit device override for testing."""
        cls._device_override = device_str

    @classmethod
    def reset_state(cls) -> None:
        """Resets the health tracking state (useful for tests or explicit re-checks)."""
        with cls._lock:
            cls._cuda_healthy = None
            cls._cuda_unhealthy_reason = None
            cls._device_override = None
