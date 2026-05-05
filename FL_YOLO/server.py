"""서버 쪽 labeled 학습 코드.
FedSTO 논문에서 서버가 labeled data로 supervised learning을 하는 부분 구현.
- Server 클래스: 모델과 labeled data loader를 받아서 supervised learning step을 수행하는 클래스
- warmup 메서드: 라운드 시작 전에 warmup_rounds epochs 동안 labeled data로 모델을 사전 학습하는 메서드
- update 메서드: 매 라운드마다 server_steps 만큼 labeled data로 supervised step을 수행하는 메서드
- supervised step에서는 모델의 supervised_loss 메서드를 호출하여 손실을 계산하고, 옵티마이저로 역전파하여 모델을 업데이트
-orthogonal regularization도 옵션으로 포함되어 있음 
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
      
        self.model.train()
        self.reset_optimizer()
        opt = self._ensure_optimizer()
        total = 0.0
        n_steps = 0
        for rnd in range(self.cfg.warmup_rounds):
            epoch_loss = 0.0
            epoch_steps = 0
            for batch in self.loader:
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
                opt.zero_grad()
                loss.backward()
                opt.step()
                step_loss = float(loss.detach())
                total += step_loss
                epoch_loss += step_loss
                n_steps += 1
                epoch_steps += 1
                if epoch_steps % 100 == 0:
                    print(f"  [warmup] epoch {rnd+1}/{self.cfg.warmup_rounds}  "
                          f"step {epoch_steps}  loss={step_loss:.4f}", flush=True)
            avg = epoch_loss / max(epoch_steps, 1)
            print(f"  [warmup] epoch {rnd+1}/{self.cfg.warmup_rounds} done  "
                  f"avg_loss={avg:.4f}  steps={epoch_steps}", flush=True)
       
        self.reset_optimizer()
        return total / max(n_steps, 1)

    def update(self, use_ortho: bool) -> float: #서버와 클라이언트의 스텝 수
        
        self.model.train()
        opt = self._ensure_optimizer()
        total = 0.0
        n_steps = 0
        if self.cfg.server_epoch:
            # 논문 기본값: 레이블 데이터 전체를 1 epoch 학습
            for batch in self.loader:
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
                total += float(loss.detach())
                n_steps += 1
        else:
            for _ in range(self.cfg.server_steps):
                total += self._supervised_step(opt, use_ortho=use_ortho)
                n_steps += 1
        return total / max(n_steps, 1)
