"""클라이언트에서 올라온 모델 가충치를 합치는 파일. 
여러 클리이언트 모델을 어떻게 평균낼 것인가에 대한 방법
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
            # 실수형이 아닌 값(예: num_batches_tracked)은 첫 번째 클라이언트 값을 사용한다.
            avg[k] = v_ref.clone()
    return avg
