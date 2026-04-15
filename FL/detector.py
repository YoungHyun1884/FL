"""Detector interface for FedSTO.

To plug in YOLOv5/YOLOv8/Faster-RCNN/etc., subclass `BaseDetector` and:
  1. Place backbone submodules under names listed in `BACKBONE_PREFIXES`.
  2. Place neck/head submodules under names listed in `NON_BACKBONE_PREFIXES`.
  3. Implement `supervised_loss(images, targets) -> dict[str, Tensor]`.
  4. Implement `unsupervised_loss(images, teacher) -> dict[str, Tensor]`
     using `teacher` (another BaseDetector instance, typically the local EMA)
     to generate pseudo labels internally.

The framework relies only on these two loss methods + parameter splitting,
so it stays detector-agnostic.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDetector(nn.Module, ABC):
    # Subclasses override these to declare parameter partitioning.
    BACKBONE_PREFIXES: Tuple[str, ...] = ("backbone",)
    NON_BACKBONE_PREFIXES: Tuple[str, ...] = ("neck", "head")

    # ---- loss API (subclass implements) ---------------------------------
    @abstractmethod
    def supervised_loss(self, images: torch.Tensor, targets) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def unsupervised_loss(
        self, images: torch.Tensor, teacher: "BaseDetector"
    ) -> Dict[str, torch.Tensor]:
        ...

    # ---- parameter / state partitioning ---------------------------------
    def _is_backbone(self, name: str) -> bool:
        return any(name.startswith(p + ".") or name == p for p in self.BACKBONE_PREFIXES)

    def _is_non_backbone(self, name: str) -> bool:
        return any(
            name.startswith(p + ".") or name == p for p in self.NON_BACKBONE_PREFIXES
        )

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        for n, p in self.named_parameters():
            if self._is_backbone(n):
                yield p

    def non_backbone_parameters(self) -> Iterable[nn.Parameter]:
        for n, p in self.named_parameters():
            if self._is_non_backbone(n):
                yield p

    def non_backbone_weight_matrices(self) -> Iterable[torch.Tensor]:
        """Conv/Linear weights in non-backbone for orthogonal regularization."""
        for module_name in self.NON_BACKBONE_PREFIXES:
            module = getattr(self, module_name, None)
            if module is None:
                continue
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    yield m.weight

    def backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        full = self.state_dict()
        return {k: v for k, v in full.items() if self._is_backbone(k)}

    def load_backbone_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        # strict=False because we're only loading a subset
        missing = self.load_state_dict(sd, strict=False)
        # Sanity: nothing unexpected (all keys in sd should be in self)
        if missing.unexpected_keys:
            raise RuntimeError(f"Unexpected keys in backbone sd: {missing.unexpected_keys}")

    # ---- threshold control (Epoch Adaptor) --------------------------------
    def set_thresholds(self, tau_low: float, tau_high: float) -> None:
        """Update pseudo-label thresholds (called by Epoch Adaptor each round)."""
        self.tau_low = tau_low
        self.tau_high = tau_high

    # ---- freezing helpers ------------------------------------------------
    def freeze_non_backbone(self) -> None:
        for n, p in self.named_parameters():
            if self._is_non_backbone(n):
                p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)


# =====================================================================
# Toy detector for framework-level testing on synthetic data.
# Replace with a real detector (YOLO etc.) in production.
# =====================================================================
class DummyDetector(BaseDetector):
    """Classifies presence class + regresses a 2D center. Not a real detector."""

    def __init__(self, num_classes: int = 5, in_channels: int = 3,
                 tau_low: float = 0.1, tau_high: float = 0.6):
        super().__init__()
        self.num_classes = num_classes
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.last_num_pseudo = 0
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.neck = nn.Sequential( 
            nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU(), # 32*4*4=512 -> 64로 차원 축소하는 간단한 neck (백본에서 추출된 특징을 더 작은 차원으로 변환)
        )
        self.head = nn.Linear(64, num_classes + 2)  # 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.neck(self.backbone(x)))

    def supervised_loss(self, images, targets):
        out = self.forward(images)
        cls_logits = out[:, : self.num_classes]
        xy = out[:, self.num_classes :]
        cls_loss = F.cross_entropy(cls_logits, targets["cls"])
        reg_loss = F.smooth_l1_loss(xy, targets["xy"])
        return {"cls": cls_loss, "reg": reg_loss}

    def unsupervised_loss(self, images, teacher):
        # Teacher (local EMA) generates pseudo labels
        teacher.eval()
        with torch.no_grad():
            t_out = teacher(images)
            t_cls_probs = t_out[:, : self.num_classes].softmax(-1)
            t_xy = t_out[:, self.num_classes :]
            conf, pseudo_cls = t_cls_probs.max(-1)

        # 3-tier Pseudo Label Assigner (Efficient Teacher)
        hard_mask = conf >= self.tau_high
        soft_mask = (conf >= self.tau_low) & (conf < self.tau_high)
        device = images.device
        self.last_num_pseudo = int(hard_mask.sum() + soft_mask.sum())

        if hard_mask.sum() == 0 and soft_mask.sum() == 0:
            z = torch.zeros((), device=device)
            return {"cls": z, "reg": z}

        s_out = self.forward(images)
        s_cls = s_out[:, : self.num_classes]
        s_xy = s_out[:, self.num_classes :]

        cls_loss = torch.zeros((), device=device)
        reg_loss = torch.zeros((), device=device)

        # Hard pseudo labels — full weight
        if hard_mask.sum() > 0:
            cls_loss = cls_loss + F.cross_entropy(s_cls[hard_mask], pseudo_cls[hard_mask])
            reg_loss = reg_loss + F.smooth_l1_loss(s_xy[hard_mask], t_xy[hard_mask])

        # Soft pseudo labels — weighted by mean teacher confidence
        if soft_mask.sum() > 0:
            soft_w = conf[soft_mask].mean()
            cls_loss = cls_loss + soft_w * F.cross_entropy(s_cls[soft_mask], pseudo_cls[soft_mask])
            reg_loss = reg_loss + soft_w * F.smooth_l1_loss(s_xy[soft_mask], t_xy[soft_mask])

        return {"cls": cls_loss, "reg": reg_loss}
