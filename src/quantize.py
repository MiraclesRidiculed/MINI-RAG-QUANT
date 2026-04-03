import numpy as np
import torch


def floating_point_weight_bytes(state_dict: dict) -> int:
    """Bytes if all floating-point tensors are stored in their native dtype (e.g. FP32 on disk/in RAM)."""
    total = 0
    for t in state_dict.values():
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            total += t.numel() * t.element_size()
    return total


def estimated_int8_symmetric_per_tensor_bytes(state_dict: dict) -> int:
    """Rough storage for per-tensor symmetric int8 weights + one float32 scale per tensor (see quantize_tensor)."""
    total = 0
    for t in state_dict.values():
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            total += int(t.numel())  # int8 payload
            total += 4  # float32 scale
    return total


def quantize_tensor(t):
    x = t.detach().numpy()
    m = abs(x).max()
    scale = float(m) / 127.0 if m > 0 else 1.0
    q = (x / scale).astype(np.int8)
    return q, scale


def quantize(v):
    return quantize_tensor(v)


def dequantize(q, scale):
    x = q.astype(np.float32) * scale
    return torch.from_numpy(x)
