"""클라이언트에서 올라온 모델 가충치를 합치는 파일. 
여러 클리이언트 모델을 어떻게 평균낼 것인가에 대한 방법
"""
from __future__ import annotations
from typing import Dict, List
import torch


def _is_bn_running_stat(key: str) -> bool:
    """state_dict 키가 BN running stat인지 확인한다."""
    return "running_mean" in key or "running_var" in key or "num_batches_tracked" in key


def fedavg(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
    exclude_bn: bool = False,
) -> Dict[str, torch.Tensor]:
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    total = float(sum(weights))
    assert total > 0, "Aggregation weights sum to zero"
    norm_w = [w / total for w in weights]

    avg: Dict[str, torch.Tensor] = {}
    ref = state_dicts[0]
    for k, v_ref in ref.items():
        if exclude_bn and _is_bn_running_stat(k):
            
            continue
        if v_ref.dtype.is_floating_point:
            acc = torch.zeros_like(v_ref, dtype=torch.float32)
            for sd, w in zip(state_dicts, norm_w):
                acc += sd[k].detach().to(torch.float32) * w
            avg[k] = acc.to(v_ref.dtype)
        else:
            avg[k] = v_ref.clone()
    return avg
