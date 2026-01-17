"""
Device detection utilities for PyTorch.

Provides unified device selection across all models.
"""

import torch


def get_best_device():
    """
    Detect and return the best available device for PyTorch.

    Priority:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info():
    """
    Get detailed device information.

    Returns:
        dict: Device information including name, type, and memory if available.
    """
    device = get_best_device()
    info = {
        "device": str(device),
        "type": device.type,
    }

    if device.type == "cuda":
        info["cuda_name"] = torch.cuda.get_device_name(0)
        info["memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["memory_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU"
    else:
        info["name"] = "CPU"

    return info


def print_device_info():
    """Print device information to console."""
    info = get_device_info()
    print(f"Device: {info['device']}")
    print(f"  Type: {info['type']}")
    if "cuda_name" in info:
        print(f"  Name: {info['cuda_name']}")
        print(f"  Memory: {info['memory_total_gb']:.1f} GB")
    elif "name" in info:
        print(f"  Name: {info['name']}")
