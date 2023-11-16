import torch
import numpy as np
from typing import Union

def np2th(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray.detach().cpu()
    elif isinstance(ndarray, np.ndarray):
        return torch.tensor(ndarray).float()
    else:
        raise ValueError("Input should be either torch.Tensor or np.ndarray")

def normalize_pc(p, method: str="sphere"):
    if method == "sphere":
        return _to_unit_sphere(p).numpy()
    elif method == "cube":
        return _to_unit_cube(p).numpy()
    else:
        raise AssertionError

def _to_unit_sphere(pc: Union[np.ndarray, torch.Tensor]):
    """
    pc: [B,N,3] or [N,3]
    """
    pc = np2th(pc)
    shapes = pc.shape
    N = shapes[-2]
    pc = pc.reshape(-1, N, 3)
    m = pc.mean(1, keepdim=True)
    pc = pc - m
    s = torch.max(torch.sqrt(torch.sum(pc**2, -1, keepdim=True)), 1, keepdim=True)[0]
    pc = pc / s
    pc = pc.reshape(shapes)
    return pc

def _to_unit_cube(pc: Union[np.ndarray, torch.Tensor]):
    """
    pc: [B,N,3] or [N,3]
    """
    pc = np2th(pc)
    shapes = pc.shape
    N = shapes[-2]
    pc = pc.reshape(-1,N,3)
    max_vals = pc.max(1, keepdim=True)[0] #[B,1,3]
    min_vals = pc.min(1,keepdim=True)[0] #[B,1,3]
    max_range = (max_vals - min_vals).max(-1)[0] / 2 #[B,1]
    center = (max_vals + min_vals) / 2 #[B,1,3]

    pc = pc - center
    pc = pc / max_range[..., None]
    pc = pc.reshape(shapes)
    return pc

def get_unit_cube_transform(pc):
    pc = np2th(pc)
    shapes = pc.shape
    N = shapes[-2]
    pc = pc.reshape(-1,N,3)
    max_vals = pc.max(1, keepdim=True)[0] #[B,1,3]
    min_vals = pc.min(1,keepdim=True)[0] #[B,1,3]
    max_range = (max_vals - min_vals).max(-1)[0] / 2 #[B,1]
    center = (max_vals + min_vals) / 2 #[B,1,3]
    return center, max_range
