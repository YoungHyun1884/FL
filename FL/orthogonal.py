"""페이즈2에서 넥과 헤드 쪽 가중치에 정규화를 걸 때 사용
"""
from __future__ import annotations
from typing import Iterable
import torch


def _spectral_norm(mat: torch.Tensor, n_iters: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """power iteration으로 가장 큰 특이값을 근사한다."""
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
    """각 가중치에 대해 sigma(WW^T - I) + sigma(W^TW - I)를 합산한다."""
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
        # 정규화할 2차원 이상 가중치가 없을 때
        # autograd에 무리 없이 참여할 수 있는 스칼라 텐서를 반환한다.
        return torch.zeros((), requires_grad=False)
    return total
