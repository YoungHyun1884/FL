"""탐지기가 공통 인터페이스를 정의.
supervised_loss: labeled data에서의 손실 계산
unsupervised_loss: 선생생 모델로부터 수두을 생성하여 unlabeled data에서의 손실 계산
backbone_parameters / non_backbone_parameters: 연합 학습에서 백본과 비백본 매개변수 구분
set_thresholds: Epoch Adaptor가 라운마다 pseudo-label 임계값 업데이트를 위해 호출
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDetector(nn.Module, ABC):
    # 하위 클래스가 파라미터 분할 규칙을 선언할 때 이 값을 재정의한다.
    BACKBONE_PREFIXES: Tuple[str, ...] = ("backbone",)
    NON_BACKBONE_PREFIXES: Tuple[str, ...] = ("neck", "head")

    # ---- loss API (하위 클래스 구현) ---------------------------------
    @abstractmethod
    def supervised_loss(self, images: torch.Tensor, targets) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def unsupervised_loss(
        self, images: torch.Tensor, teacher: "BaseDetector"
    ) -> Dict[str, torch.Tensor]:
        ...

    # ---- 파라미터 / 상태 분할 ---------------------------------
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
        """직교 정규화에 사용할 non-backbone의 Conv/Linear 가중치를 반환한다."""
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
        # 일부 파라미터만 불러오므로 strict=False를 사용한다.
        missing = self.load_state_dict(sd, strict=False)
        # 점검: sd의 모든 키는 현재 모델 안에 존재해야 한다.
        if missing.unexpected_keys:
            raise RuntimeError(f"Unexpected keys in backbone sd: {missing.unexpected_keys}")

    # ---- 임계값 제어 (Epoch Adaptor) --------------------------------
    def set_thresholds(self, tau_low: float, tau_high: float) -> None:
        """가짜 라벨 임계값을 갱신한다. 각 라운드마다 Epoch Adaptor가 호출한다."""
        self.tau_low = tau_low
        self.tau_high = tau_high

    # ---- 동결 보조 함수 ------------------------------------------------
    def freeze_non_backbone(self) -> None:
        for n, p in self.named_parameters():
            if self._is_non_backbone(n):
                p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)


# =====================================================================
# 프레임워크 수준 테스트용 장난감 탐지기.
# 실제 사용 시에는 진짜 탐지기(YOLO 등)로 교체한다.
# =====================================================================
class DummyDetector(BaseDetector):
    """클래스 존재 여부를 분류하고 2D 중심을 회귀하는 간단한 예제 탐지기다."""

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
        # 선생 모델(local EMA)이 가짜 라벨을 만든다.
        teacher.eval()
        with torch.no_grad():
            t_out = teacher(images)
            t_cls_probs = t_out[:, : self.num_classes].softmax(-1)
            t_xy = t_out[:, self.num_classes :]
            conf, pseudo_cls = t_cls_probs.max(-1)

        # 3단계 가짜 라벨 할당기(Efficient Teacher)
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

        # hard pseudo label은 전체 가중치로 반영한다.
        if hard_mask.sum() > 0:
            cls_loss = cls_loss + F.cross_entropy(s_cls[hard_mask], pseudo_cls[hard_mask])
            reg_loss = reg_loss + F.smooth_l1_loss(s_xy[hard_mask], t_xy[hard_mask])

        # soft pseudo label은 teacher 평균 신뢰도로 가중한다.
        if soft_mask.sum() > 0:
            soft_w = conf[soft_mask].mean()
            cls_loss = cls_loss + soft_w * F.cross_entropy(s_cls[soft_mask], pseudo_cls[soft_mask])
            reg_loss = reg_loss + soft_w * F.smooth_l1_loss(s_xy[soft_mask], t_xy[soft_mask])

        return {"cls": cls_loss, "reg": reg_loss}
