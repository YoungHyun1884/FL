"""Orthogonal regularization (SRIP) from Kim & Yun (2022), Eq. used in FedSTO.

    R(theta) = sum_theta [ sigma(theta^T theta - I) + sigma(theta theta^T - I) ]

where sigma(.) is the spectral norm, approximated via one (or more) step of
power iteration. Applied to non-backbone weight matrices only.

Convolutional weights are reshaped to 2D as (out_channels, in_channels * kH * kW).
"""
from __future__ import annotations
from typing import Iterable
import torch


def _spectral_norm(mat: torch.Tensor, n_iters: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """Approximate largest singular value via power iteration."""
    u = torch.randn(mat.size(0), device=mat.device, dtype=mat.dtype)
    u = u / (u.norm() + eps)
    for _ in range(n_iters):
        v = mat.t() @ u
        v = v / (v.norm() + eps)
        u = mat @ v
        u = u / (u.norm() + eps)
    sigma = (u @ mat @ v)
    return sigma.abs()


def srip_penalty(
    weights: Iterable[torch.Tensor], n_iters: int = 1
) -> torch.Tensor:
    """Sum over weights of sigma(WW^T - I) + sigma(W^TW - I)."""
    total = None
    for w in weights:
        if w.ndim < 2:
            continue
        wm = w.reshape(w.size(0), -1)
        wwt = wm @ wm.t()
        wtw = wm.t() @ wm
        I1 = torch.eye(wwt.size(0), device=w.device, dtype=w.dtype)
        I2 = torch.eye(wtw.size(0), device=w.device, dtype=w.dtype)
        s1 = _spectral_norm(wwt - I1, n_iters=n_iters)
        s2 = _spectral_norm(wtw - I2, n_iters=n_iters)
        term = s1 + s2
        total = term if total is None else total + term
    if total is None:
        # no 2D+ weights to regularize
        # return a scalar tensor that participates in autograd cleanly
        return torch.zeros((), requires_grad=False)
    return total
