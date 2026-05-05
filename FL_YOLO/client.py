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
    def _sync_from_global(self, global_model: BaseDetector) -> None: #client는 이전 라운드의 자신의 로컬 모델을 이어서 학습 x 매 라운드 글로벌 기준점에서 다시 시작
        self.model.load_state_dict(global_model.state_dict())
        if self.local_ema is None: #첫 라운드에 teacher가 없으니 현재 student 모델을 복사
            self.local_ema = LocalEMA(self.model, decay=self.cfg.ema_decay)
            self.local_ema.to(self.device)
        elif self.cfg.reset_ema_each_round:
            self.local_ema.reset_from(self.model)

    def train_phase1(
        self, global_model: BaseDetector,
        tau_low: float | None = None, tau_high: float | None = None,
    ) -> Dict:
        self._sync_from_global(global_model) #현재 클라이언트 모델을 글로벌 모델과 맞춤
        if tau_low is not None and tau_high is not None:
            self.model.set_thresholds(tau_low, tau_high)
        self.model.set_pseudo_config( #hard_only인지, soft psedudo에 추가 weight를 얼마나 줄지 결정
            use_soft_pseudo=not self.cfg.phase1_hard_only,
            soft_pseudo_weight=self.cfg.phase1_soft_weight,
        )
        self.model.train()
        self.model.freeze_non_backbone()

        opt = torch.optim.SGD(
            [p for p in self.model.backbone_parameters() if p.requires_grad], #백본 파라미터만 업데이트
            lr=self.cfg.client_opt.lr * self.cfg.phase1_client_lr_scale, #lr에 phase1_client_lr_scale 곱해서 Phase 1에서는 더 작은 lr로 backbone 업데이트
            momentum=self.cfg.client_opt.momentum,
            weight_decay=self.cfg.client_opt.weight_decay,
        )

        lam = self.cfg.unsup_loss_weight
        total_loss = 0.0
        total_pseudo = 0
        n_steps = 0

        for _epoch in range(self.cfg.local_epochs):
            for bi, batch in enumerate(self.loader):
                if (
                    self.cfg.phase1_max_batches_per_epoch is not None
                    and bi >= self.cfg.phase1_max_batches_per_epoch
                ):
                    break
                images = batch["images"].to(self.device) #클라리언트 lodaer에서 꺼낸 자기 로컬 이미지
                loss_dict = self.model.unsupervised_loss(images, self.local_ema.ema) #self.model : 학생 , self.local_ema.ema : 선생 => teacher가 수도라벨을 만들고 student가 그 수도 라벨에 맞춰 학습하도록 loos 계산산
                loss = lam * sum(loss_dict.values())
                total_pseudo += int(getattr(self.model, "last_num_pseudo", 0))
                n_steps += 1
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    continue
                opt.zero_grad() #학생만 업데이트
                loss.backward() #학생만 업데이트
                opt.step() #학생만 업데이트
                self.local_ema.update(self.model) #선생이 학생을 ema로 따라가게 함.
                total_loss += float(loss.detach())

        self.model.unfreeze_all()
        return { #클라이언트는 모델 전체가 아리나 bacbone_state_dict만 반환, 나머지는 서버에서 업데이트 안하니 반환할 필요 없음
            "backbone_state_dict": {
                k: v.detach().cpu() for k, v in self.model.backbone_state_dict().items()
            },
            "num_samples": self.num_samples, #fedavg 가중치
            "loss": total_loss / max(n_steps, 1), 
            "num_pseudo": total_pseudo, #수도라벨 생성량
        }

    def train_phase2(
        self, global_model: BaseDetector,
        tau_low: float | None = None, tau_high: float | None = None,
    ) -> Dict:
        self._sync_from_global(global_model)
        if tau_low is not None and tau_high is not None:
            self.model.set_thresholds(tau_low, tau_high)
        self.model.set_pseudo_config(use_soft_pseudo=True, soft_pseudo_weight=1.0)
        self.model.train()
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
                loss_dict = self.model.unsupervised_loss(images, self.local_ema.ema)
                task_loss = lam * sum(loss_dict.values())
                num_pseudo = int(getattr(self.model, "last_num_pseudo", 0))
                total_pseudo += num_pseudo
                if num_pseudo == 0:
                  
                    n_steps += 1
                    continue
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
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "num_samples": self.num_samples,
            "loss": total_loss / max(n_steps, 1),
            "ortho": total_ortho / max(n_steps, 1),
            "num_pseudo": total_pseudo,
        }
