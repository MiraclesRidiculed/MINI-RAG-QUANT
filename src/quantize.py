import numpy as np
import torch


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
