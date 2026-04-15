"""Local Exponential Moving Average teacher model.

Per FedSTO paper (Appendix H.2): each client maintains a local EMA model used
as pseudo labeler. At the start of every round the EMA weights are re-initialized
from the freshly broadcast global model, then updated after each local step as:

    W_EMA <- alpha * W_EMA + (1 - alpha) * W_student
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
        """Re-initialize EMA weights to match the given model (called at round start)."""
        self.ema.load_state_dict(model.state_dict())

    def to(self, device) -> "LocalEMA":
        self.ema.to(device)
        return self
