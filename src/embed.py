import torch

def embed(text):
    torch.manual_seed(len(text))  # deterministic
    return torch.randn(8)