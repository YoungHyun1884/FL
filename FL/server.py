"""FedSTO server.
  - client는 unlabeled 데이터로 local adaptation을 담당한다.
  - server는 labeled 데이터로 global drift를 보정하는 역할을 맡는다.
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
        """옵티마이저를 다시 만든다. 단계 사이에서 모멘텀을 초기화할 때 사용한다."""
        self._opt = None

    def _supervised_step(self, opt, use_ortho: bool) -> float:
        # 집계 후 서버는 레이블 배치로 다시 한 번 정답 기반 보정을 한다.
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
            # 2단계에서는 클라이언트 목적함수와 맞추기 위해 서버도 같은 정규화를 사용한다.
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
        """사전학습 단계에서 레이블 데이터를 warmup_rounds 동안 지도학습한다."""
        # 연합학습 전에 서버가 먼저 레이블 데이터로 detector를 안정화한다.
        self.model.train()
        self.reset_optimizer()
        opt = self._ensure_optimizer()
        total = 0.0
        n_steps = 0
        for _rnd in range(self.cfg.warmup_rounds):
            for batch in self.loader:
                batch_data = batch
                # 레이블 데이터는 이미 collate 단계에서 YOLO 타깃 형식으로 정리돼 있다.
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
        # warmup에서 쌓인 모멘텀을 넘기지 않도록 optimizer를 다시 초기화한다.
        self.reset_optimizer()
        return total / max(n_steps, 1)

    def update(self, use_ortho: bool) -> float:
        # 각 round에서 집계 직후 server_steps만큼 지도 보정을 수행한다.
        self.model.train()
        opt = self._ensure_optimizer()
        total = 0.0
        for _ in range(self.cfg.server_steps):
            total += self._supervised_step(opt, use_ortho=use_ortho)
        return total / max(self.cfg.server_steps, 1)
