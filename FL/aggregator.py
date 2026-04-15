"""Weighted FedAvg aggregation over state dicts.

Supports two modes:
  - Full state dict aggregation (Phase 2)
  - Backbone-only aggregation (Phase 1): filter keys before averaging
"""
from __future__ import annotations
from typing import Dict, List
import torch


def fedavg(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    total = float(sum(weights))
    assert total > 0, "Aggregation weights sum to zero"
    norm_w = [w / total for w in weights]

    avg: Dict[str, torch.Tensor] = {}
    ref = state_dicts[0]
    for k, v_ref in ref.items():
        if v_ref.dtype.is_floating_point:
            acc = torch.zeros_like(v_ref, dtype=torch.float32)
            for sd, w in zip(state_dicts, norm_w):
                acc += sd[k].detach().to(torch.float32) * w
            avg[k] = acc.to(v_ref.dtype)
        else:
            # Non-float (e.g. num_batches_tracked): take the first client's copy
            avg[k] = v_ref.clone()
    return avg
