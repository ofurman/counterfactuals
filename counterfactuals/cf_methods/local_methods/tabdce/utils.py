from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1))).expand(x_shape)


def index_to_log_onehot(x: torch.Tensor, num_classes: torch.Tensor) -> torch.Tensor:
    B = x.size(0)
    m = num_classes.numel()
    splits = num_classes.tolist()
    xs = x.long().split(1, dim=1)
    outs = []
    for i, K in enumerate(splits):
        oh = F.one_hot(xs[i].squeeze(1), num_classes=K).float()
        outs.append(torch.log(oh.clamp(min=1e-30)))
    return torch.cat(outs, dim=1)


def log_onehot_to_index(log_x: torch.Tensor, num_classes: torch.Tensor) -> torch.Tensor:
    B = log_x.size(0)
    splits = num_classes.tolist()
    offs = [0]
    for K in splits: offs.append(offs[-1] + K)
    idx = []
    for i, K in enumerate(splits):
        sl = slice(offs[i], offs[i+1])
        idx.append(log_x[:, sl].argmax(dim=1, keepdim=True))
    return torch.cat(idx, dim=1)


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=tuple(range(1, x.ndim)))


@dataclass
class DiffusionSchedule:
    T: int
    betas: torch.Tensor

    @staticmethod
    def from_name(name: str, T: int, device: torch.device, dtype: torch.dtype = torch.float32) -> "DiffusionSchedule":
        if name == "linear":
            scale = 1000.0 / T
            beta_start = scale * 1e-4
            beta_end = scale * 2e-2
            betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype)
        elif name == "cosine":
            s = 0.008
            steps = torch.arange(T+1, device=device, dtype=dtype)
            f = torch.cos(((steps / T + s) / (1 + s)) * torch.pi / 2) ** 2
            alphas_bar = f / f[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1]).clamp(min=1e-5, max=0.9999)
        else:
            raise ValueError(f"Unknown schedule: {name}")
        return DiffusionSchedule(T=T, betas=betas)


def postprocess_cf(
    cf_model: np.ndarray, 
    qt=None,         
    ohe=None,          
    clip_qt: float = 3.5,   
):
    N, D = cf_model.shape

    n_num = qt.n_features_in_ if qt is not None else 0

    cf_num_tr = cf_model[:, :n_num]        
    cf_cat_log = cf_model[:, n_num:]        
    if qt is not None and n_num > 0:
        cf_num_tr = np.clip(cf_num_tr, -clip_qt, clip_qt)
        cf_num = qt.inverse_transform(cf_num_tr)
    else:
        cf_num = np.zeros((N, 0), dtype=np.float32)

    cf_cat_idx = None
    if cf_cat_log.size > 0 and ohe is not None:
        cat_idx_list = []
        start = 0
        for cats in ohe.categories_:
            K = len(cats)
            blk = cf_cat_log[:, start:start+K]  
            idx = blk.argmax(axis=1)
            cat_idx_list.append(idx)
            start += K
        cf_cat_idx = np.stack(cat_idx_list, axis=1)   

    return cf_num, cf_cat_idx