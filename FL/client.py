"""클라이언트 쪽 로컬 학습 하는 파일
global model 받아오기 -> local EMA teacher 준비 -> unlabeled 이미지로 pseudo label 학습 -> 업데이트 결과 반환
Phase 1에서는 backbone만, Phase 2에서는 전체 파라미터를 업데이트.
"""
from __future__ import annotations
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .config import FedSTOConfig
from .detector import BaseDetector
from .ema import LocalEMA
from .orthogonal import srip_penalty


class Client:
    def __init__(
        self,
        client_id: int,
        model: BaseDetector,
        loader: DataLoader,
        cfg: FedSTOConfig,
        num_samples: int,
    ):
        self.cid = client_id
        self.model = model.to(cfg.device)
        self.loader = loader
        self.cfg = cfg
        self.num_samples = num_samples
        self.device = cfg.device
        self.local_ema: LocalEMA | None = None

    # -------- 라운드 시작 동기화 ------------------------------------------
    def _sync_from_global(self, global_model: BaseDetector) -> None:
        # 매 라운드 시작 시 클라이언트 student를 최신 전역 가중치로 덮어쓴다.
        self.model.load_state_dict(global_model.state_dict())
        if self.local_ema is None:
            # 첫 라운드에서는 student 복사본으로 EMA teacher를 만든다.
            self.local_ema = LocalEMA(self.model, decay=self.cfg.ema_decay)
            self.local_ema.to(self.device)
        elif self.cfg.reset_ema_each_round:
            # 설정에 따라 각 라운드 시작마다 EMA를 전역 기준으로 다시 맞춘다.
            self.local_ema.reset_from(self.model)

    # -------- 1단계: 선택적 학습 -------------------------------
    def train_phase1(
        self, global_model: BaseDetector,
        tau_low: float | None = None, tau_high: float | None = None,
    ) -> Dict:
        self._sync_from_global(global_model)
        if tau_low is not None and tau_high is not None:
            self.model.set_thresholds(tau_low, tau_high)
        self.model.train()
        # 1단계의 핵심: backbone만 로컬 적응시키고 neck/head는 고정한다.
        self.model.freeze_non_backbone()

        opt = torch.optim.SGD(
            [p for p in self.model.backbone_parameters() if p.requires_grad],
            lr=self.cfg.client_opt.lr,
            momentum=self.cfg.client_opt.momentum,
            weight_decay=self.cfg.client_opt.weight_decay,
        )

        lam = self.cfg.unsup_loss_weight
        total_loss = 0.0
        total_pseudo = 0
        n_steps = 0

        for _epoch in range(self.cfg.local_epochs):
            for batch in self.loader:
                images = batch["images"].to(self.device)
                # unlabeled 이미지 -> EMA teacher 가짜 라벨 -> student 비지도 손실
                loss_dict = self.model.unsupervised_loss(images, self.local_ema.ema)
                loss = lam * sum(loss_dict.values())
                total_pseudo += int(getattr(self.model, "last_num_pseudo", 0))
                n_steps += 1
                # 가짜 박스가 하나도 없으면 손실이 gradient를 만들지 않을 수 있다.
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
                # student가 한 스텝 좋아지면 teacher도 EMA로 천천히 따라간다.
                self.local_ema.update(self.model)
                total_loss += float(loss.detach())

        self.model.unfreeze_all()
        return {
            # 1단계에서는 backbone만 서버에 보내 선택적 집계를 수행한다.
            "backbone_state_dict": {
                k: v.detach().cpu() for k, v in self.model.backbone_state_dict().items()
            },
            "num_samples": self.num_samples,
            "loss": total_loss / max(n_steps, 1),
            "num_pseudo": total_pseudo,
        }

    # -------- 2단계: FPT + 직교 보정 ------------------
    def train_phase2(
        self, global_model: BaseDetector,
        tau_low: float | None = None, tau_high: float | None = None,
    ) -> Dict:
        self._sync_from_global(global_model)
        if tau_low is not None and tau_high is not None:
            self.model.set_thresholds(tau_low, tau_high)
        self.model.train()
        # 2단계에서는 backbone + neck/head 전체를 함께 업데이트한다.
        self.model.unfreeze_all()

        opt = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.client_opt.lr,
            momentum=self.cfg.client_opt.momentum,
            weight_decay=self.cfg.client_opt.weight_decay,
        )

        lam = self.cfg.unsup_loss_weight
        total_loss = 0.0
        total_ortho = 0.0
        total_pseudo = 0
        n_steps = 0

        for _epoch in range(self.cfg.local_epochs):
            for batch in self.loader:
                images = batch["images"].to(self.device)
                # 여전히 unlabeled 기반 학습이지만 이제는 전체 파라미터를 학습한다.
                loss_dict = self.model.unsupervised_loss(images, self.local_ema.ema)
                task_loss = lam * sum(loss_dict.values())
                total_pseudo += int(getattr(self.model, "last_num_pseudo", 0))
                # 논문 설정에 맞춰 neck/head 쪽 가중치에 직교 정규화를 추가한다.
                ortho = srip_penalty(
                    self.model.non_backbone_weight_matrices(),
                    n_iters=self.cfg.ortho_power_iters,
                )
                loss = task_loss + self.cfg.ortho_lambda * ortho
                n_steps += 1
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.local_ema.update(self.model)
                total_loss += float(task_loss.detach() if torch.is_tensor(task_loss) else 0.0)
                total_ortho += float(ortho.detach() if torch.is_tensor(ortho) else 0.0)

        return {
            # 2단계에서는 전체 state_dict를 FedAvg 대상으로 서버에 반환한다.
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "num_samples": self.num_samples,
            "loss": total_loss / max(n_steps, 1),
            "ortho": total_ortho / max(n_steps, 1),
            "num_pseudo": total_pseudo,
        }
