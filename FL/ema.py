"""local EMA teacher을 관리.
student 모델을 지수이동평균으로 따라가는 teacher를 만들고 라운드마다 reset/update하는 역할.
수두라벨 teacher 관리 전담 파일
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn


class LocalEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    @torch.no_grad()
    def reset_from(self, model: nn.Module) -> None:
        """주어진 모델과 같아지도록 EMA 가중치를 다시 초기화한다. 라운드 시작 시 호출한다."""
        self.ema.load_state_dict(model.state_dict())

    def to(self, device) -> "LocalEMA":
        self.ema.to(device)
        return self
