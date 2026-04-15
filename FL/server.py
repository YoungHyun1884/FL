"""FedSTO server.

The server holds the labeled dataset. Responsibilities:
  - Warmup: supervised pretraining on labeled data before FL starts.
  - Post-aggregation update (Phase 1 & 2): after receiving the aggregated client
    update, fine-tune on labeled data. In Phase 2 the SRIP penalty is added
    to the non-backbone weights, matching the clients' objective.
"""
from __future__ import annotations
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from .config import FedSTOConfig
from .detector import BaseDetector
from .orthogonal import srip_penalty


def _infinite(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


class Server:
    def __init__(
        self,
        model: BaseDetector,
        loader: DataLoader,
        cfg: FedSTOConfig,
    ):
        self.model = model.to(cfg.device)
        self.loader = loader
        self.cfg = cfg
        self.device = cfg.device
        self._iter = None
        self._opt: torch.optim.Optimizer | None = None

    def _next_batch(self):
        if self._iter is None:
            self._iter = _infinite(self.loader)
        return next(self._iter)

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self._opt is None:
            self._opt = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.server_opt.lr,
                momentum=self.cfg.server_opt.momentum,
                weight_decay=self.cfg.server_opt.weight_decay,
            )
        return self._opt

    def reset_optimizer(self) -> None:
        """Re-create the optimizer (call between phases to reset momentum)."""
        self._opt = None

    def _supervised_step(self, opt, use_ortho: bool) -> float:
        batch = self._next_batch()
        images = batch["images"].to(self.device)
        raw_targets = batch["targets"]
        if isinstance(raw_targets, dict):
            targets = {k: v.to(self.device) for k, v in raw_targets.items()}
        elif torch.is_tensor(raw_targets):
            targets = raw_targets.to(self.device)
        else:
            targets = raw_targets
        loss_dict = self.model.supervised_loss(images, targets)
        loss = sum(loss_dict.values())
        if use_ortho:
            ortho = srip_penalty(
                self.model.non_backbone_weight_matrices(),
                n_iters=self.cfg.ortho_power_iters,
            )
            loss = loss + self.cfg.ortho_lambda * ortho
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.detach())

    def warmup(self) -> float:
        """Warmup: supervised pretraining for warmup_rounds epochs over labeled data."""
        self.model.train()
        self.reset_optimizer()
        opt = self._ensure_optimizer()
        total = 0.0
        n_steps = 0
        for _rnd in range(self.cfg.warmup_rounds):
            for batch in self.loader:
                batch_data = batch
                # push to device inline
                images = batch_data["images"].to(self.device)
                raw_targets = batch_data["targets"]
                if isinstance(raw_targets, dict):
                    targets = {k: v.to(self.device) for k, v in raw_targets.items()}
                elif torch.is_tensor(raw_targets):
                    targets = raw_targets.to(self.device)
                else:
                    targets = raw_targets
                loss_dict = self.model.supervised_loss(images, targets)
                loss = sum(loss_dict.values())
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach())
                n_steps += 1
        # Reset optimizer after warmup so Phase 1 starts fresh
        self.reset_optimizer()
        return total / max(n_steps, 1)

    def update(self, use_ortho: bool) -> float:
        self.model.train()
        opt = self._ensure_optimizer()
        total = 0.0
        for _ in range(self.cfg.server_steps):
            total += self._supervised_step(opt, use_ortho=use_ortho)
        return total / max(self.cfg.server_steps, 1)
