"""FedSTO client.

Each client owns an unlabeled dataset and performs either:
  - Phase 1 (Selective Training): only backbone parameters are updated; neck/head
    are frozen. Returns backbone state dict.
  - Phase 2 (Full Parameter Training + Orthogonal Enhancement): all parameters
    are updated, SRIP penalty applied to non-backbone weights. Returns full state dict.

Local EMA model is used as the pseudo labeler (Semi-Efficient Teacher style).

Following the paper, each round performs 1 local epoch (full pass over the
client's local dataset) rather than a fixed number of gradient steps.
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

    # -------- round-start sync ------------------------------------------
    def _sync_from_global(self, global_model: BaseDetector) -> None:
        self.model.load_state_dict(global_model.state_dict())
        if self.local_ema is None:
            self.local_ema = LocalEMA(self.model, decay=self.cfg.ema_decay)
            self.local_ema.to(self.device)
        elif self.cfg.reset_ema_each_round:
            self.local_ema.reset_from(self.model)

    # -------- Phase 1: Selective Training -------------------------------
    def train_phase1(
        self, global_model: BaseDetector,
        tau_low: float | None = None, tau_high: float | None = None,
    ) -> Dict:
        self._sync_from_global(global_model)
        if tau_low is not None and tau_high is not None:
            self.model.set_thresholds(tau_low, tau_high)
        self.model.train()
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
                loss_dict = self.model.unsupervised_loss(images, self.local_ema.ema)
                loss = lam * sum(loss_dict.values())
                total_pseudo += int(getattr(self.model, "last_num_pseudo", 0))
                n_steps += 1
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.local_ema.update(self.model)
                total_loss += float(loss.detach())

        self.model.unfreeze_all()
        return {
            "backbone_state_dict": {
                k: v.detach().cpu() for k, v in self.model.backbone_state_dict().items()
            },
            "num_samples": self.num_samples,
            "loss": total_loss / max(n_steps, 1),
            "num_pseudo": total_pseudo,
        }

    # -------- Phase 2: FPT with Orthogonal Enhancement ------------------
    def train_phase2(
        self, global_model: BaseDetector,
        tau_low: float | None = None, tau_high: float | None = None,
    ) -> Dict:
        self._sync_from_global(global_model)
        if tau_low is not None and tau_high is not None:
            self.model.set_thresholds(tau_low, tau_high)
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
                total_pseudo += int(getattr(self.model, "last_num_pseudo", 0))
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
