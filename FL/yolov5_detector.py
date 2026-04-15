"""YOLOv5 adapter implementing BaseDetector.

Wraps ultralytics' `yolov5` package (install via `pip install yolov5`) so that
the FedSTO framework can drive selective training, orthogonal enhancement, and
local EMA pseudo labeling on top of a real single-stage detector.

Parameter partitioning follows the YOLOv5 v6.0 architecture:
  - Backbone: layers 0-9   (CSPDarknet + SPPF)
  - Non-backbone: layers 10-24  (PANet neck + Detect head)

For semi-supervised training the teacher (local EMA) generates pseudo labels
via NMS; boxes with `conf >= tau_high` are used as hard pseudo targets, then
fed through `ComputeLoss` against the student's raw multi-scale output.

Following the FedSTO paper, COCO-pretrained weights are loaded by default and
transferred to the model (excluding the final detection head when the number
of classes differs from COCO's 80).
"""
from __future__ import annotations
import logging
import os
from typing import Dict, Iterable

import torch
import torch.nn as nn

from yolov5.models.yolo import Model as YoloModel
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.general import non_max_suppression
from yolov5.utils.downloads import attempt_download

from .detector import BaseDetector

logger = logging.getLogger(__name__)

# Mapping from model size letter to COCO pretrained weight filename
_COCO_WEIGHT_NAMES = {
    "n": "yolov5n.pt",
    "s": "yolov5s.pt",
    "m": "yolov5m.pt",
    "l": "yolov5l.pt",
    "x": "yolov5x.pt",
}


def default_hyp() -> dict:
    """Minimal YOLOv5 hyp dict needed by ComputeLoss. Values from FedSTO paper."""
    return {
        "box": 0.05,
        "cls": 0.3,           # paper: cls loss balance = 0.3
        "cls_pw": 1.0,
        "obj": 0.7,           # paper: obj loss balance = 0.7
        "obj_pw": 1.0,
        "fl_gamma": 0.0,
        "anchor_t": 4.0,
        "label_smoothing": 0.0,
    }


def _find_default_yaml(size: str = "l") -> str:
    import yolov5
    root = os.path.dirname(yolov5.__file__)
    return os.path.join(root, "models", f"yolov5{size}.yaml")


def _infer_size_from_yaml(yaml_path: str) -> str:
    """Extract model size letter (n/s/m/l/x) from yaml filename."""
    basename = os.path.basename(yaml_path)          # e.g. "yolov5s.yaml"
    for sz in ("n", "s", "m", "l", "x"):
        if f"yolov5{sz}" in basename:
            return sz
    return "l"  # fallback


def _ensure_yolov5_module_aliases() -> None:
    """Register yolov5.models/yolov5.utils under their short names.

    Ultralytics COCO checkpoints pickle ``models.yolo.Model`` etc.  When
    unpickling, Python needs ``import models`` to resolve.  The pip-installed
    ``yolov5`` package exposes these as ``yolov5.models`` / ``yolov5.utils``,
    so we create ``sys.modules`` aliases once.
    """
    import sys
    import yolov5

    for sub in ("models", "utils"):
        full = f"yolov5.{sub}"
        if full in sys.modules and sub not in sys.modules:
            sys.modules[sub] = sys.modules[full]

    # Also alias sub-submodules that the checkpoint may reference
    for key, mod in list(sys.modules.items()):
        if key.startswith("yolov5.models.") or key.startswith("yolov5.utils."):
            short = key[len("yolov5."):]        # e.g. "models.yolo"
            if short not in sys.modules:
                sys.modules[short] = mod


def _load_coco_pretrained(model: YoloModel, size: str) -> None:
    """Download COCO-pretrained weights and transfer matching parameters.

    Detection head layers whose shapes differ (due to num_classes != 80) are
    skipped so the model can be used with any number of classes.
    """
    _ensure_yolov5_module_aliases()

    weight_file = _COCO_WEIGHT_NAMES.get(size, "yolov5l.pt")
    local_path = attempt_download(weight_file)
    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    # ultralytics checkpoints store the model under 'ema' or 'model';
    # prefer 'ema' when it is not None, otherwise fall back to 'model'.
    coco_model = ckpt.get("ema") or ckpt.get("model")
    if coco_model is None:
        logger.warning("Could not find model in checkpoint %s — skipping pretrained load", weight_file)
        return

    if hasattr(coco_model, "state_dict"):
        coco_sd = coco_model.float().state_dict()
    else:
        coco_sd = coco_model

    model_sd = model.state_dict()
    transferred, skipped = 0, 0
    for k, v in coco_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            transferred += 1
        else:
            skipped += 1

    model.load_state_dict(model_sd, strict=False)
    logger.info(
        "Loaded COCO pretrained %s: %d params transferred, %d skipped (shape mismatch / head)",
        weight_file, transferred, skipped,
    )
    print(f"  [pretrained] Loaded {weight_file}: {transferred} transferred, {skipped} skipped")


class YOLOv5Detector(BaseDetector):
    """FedSTO-compatible wrapper around YOLOv5.

    Args:
        num_classes: number of object classes.
        yaml_path: path to yolov5{n,s,m,l,x}.yaml. Defaults to yolov5l.
        split_idx: layer index (inclusive-exclusive cutoff) separating backbone
                   from non-backbone. Default 10 matches YOLOv5 v6.0.
        tau_high: hard pseudo-label objectness/score cutoff.
        tau_low:  NMS conf threshold (boxes below this are never considered).
        iou_thres: NMS IoU threshold.
        hyp: YOLOv5 loss hyperparameters. Defaults from `default_hyp()`.
    """

    # These are placeholders; actual partitioning is done by overriding
    # `_is_backbone` / `_is_non_backbone` below (YOLOv5 uses flat layer indices).
    BACKBONE_PREFIXES = ()
    NON_BACKBONE_PREFIXES = ()

    _PARAM_PREFIX = "yolo.model."

    def __init__(
        self,
        num_classes: int,
        yaml_path: str | None = None,
        split_idx: int = 10,
        tau_high: float = 0.6,
        tau_low: float = 0.1,
        iou_thres: float = 0.65,
        hyp: dict | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.split_idx = split_idx
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.iou_thres = iou_thres

        yaml_path = yaml_path or _find_default_yaml("l")
        self.yolo = YoloModel(yaml_path, ch=3, nc=num_classes)

        if pretrained:
            size = _infer_size_from_yaml(yaml_path)
            _load_coco_pretrained(self.yolo, size)

        self.yolo.hyp = hyp or default_hyp()
        self.yolo.gr = 1.0  # gr = 1.0 means full iou ratio in obj loss
        # ComputeLoss caches anchors / na / nl from the model at construction time
        self.compute_loss = ComputeLoss(self.yolo)

    # ------- device transfer --------------------------------------------
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # ComputeLoss caches self.device at init time (CPU).
        # After model.to("cuda"), recreate it so anchors and device are correct.
        result.compute_loss = ComputeLoss(result.yolo)
        return result

    # ------- forward ----------------------------------------------------
    def forward(self, x: torch.Tensor):
        return self.yolo(x)

    # ------- parameter partitioning (override base class) --------------
    def _layer_idx_from_name(self, name: str) -> int | None:
        if not name.startswith(self._PARAM_PREFIX):
            return None
        rest = name[len(self._PARAM_PREFIX):]
        head = rest.split(".", 1)[0]
        return int(head) if head.isdigit() else None

    def _is_backbone(self, name: str) -> bool: # 백본 레이어 식별 : 이름에서 레이어 인덱스를 추출하여 split_idx와 비교하여 백본인지 여부를 결정
        idx = self._layer_idx_from_name(name) # 이름에서 레이어 인덱스를 추출하는 헬퍼 함수 호출
        return idx is not None and idx < self.split_idx # 추출된 인덱스가 유효하고 split_idx보다 작은 경우 백본으로 간주

    def _is_non_backbone(self, name: str) -> bool: # 논백본 레이어 식별 : 이름에서 레이어 인덱스를 추출하여 split_idx와 비교하여 논백본인지 여부를 결정
        idx = self._layer_idx_from_name(name) 
        return idx is not None and idx >= self.split_idx

    def non_backbone_weight_matrices(self) -> Iterable[torch.Tensor]:
        """Conv/Linear weights from neck + head for orthogonal regularization."""
        seq = self.yolo.model  # nn.Sequential of layers 0..N
        for layer in seq[self.split_idx:]:
            for m in layer.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    yield m.weight

    # ------- losses -----------------------------------------------------
    def supervised_loss( #서버가 labeled 데이터로 학습할 때 쓰는  함수
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Supervised YOLOv5 loss.

        Args:
            images: (B, 3, H, W)
            targets: (N, 6) [img_idx, cls, cx, cy, w, h] normalized.
        """
        self.yolo.train()
        pred = self.yolo(images)  # list of 3 tensors in train mode
        loss, _loss_items = self.compute_loss(pred, targets)
        # Single entry so framework's sum() preserves the original scale.
        return {"det": loss}

    @torch.no_grad()
    def _make_pseudo_targets_3tier( #teacher EMA 모델이 unlabed 이미지를 보고 수두 박스를 만듦.
        self, teacher: "YOLOv5Detector", images: torch.Tensor
    ) -> tuple:
        """3-tier Pseudo Label Assigner (Efficient Teacher PLA).

        Returns:
            hard_targets: (N, 6) boxes with conf >= tau_high — full weight
            soft_targets: (M, 6) boxes with tau_low <= conf < tau_high
            soft_confs:   (M,)   teacher confidence for soft targets
        """
        teacher.yolo.eval()
        out = teacher.yolo(images)
        det = out[0] if isinstance(out, (tuple, list)) else out  # (B, N, 5+nc)
        nms_out = non_max_suppression(
            det, conf_thres=self.tau_low, iou_thres=self.iou_thres
        )

        H, W = images.shape[-2:]
        hard_list, soft_list, conf_list = [], [], []
        for i, boxes in enumerate(nms_out):
            if boxes is None or len(boxes) == 0:
                continue
            # boxes: (K, 6) [x1, y1, x2, y2, conf, cls]  in pixel coords
            confs = boxes[:, 4]

            for mask, target_list in [
                (confs >= self.tau_high, hard_list),
                ((confs >= self.tau_low) & (confs < self.tau_high), soft_list),
            ]:
                sel = boxes[mask]
                if len(sel) == 0:
                    continue
                x1, y1, x2, y2 = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
                cx = ((x1 + x2) * 0.5 / W).clamp(0, 1)
                cy = ((y1 + y2) * 0.5 / H).clamp(0, 1)
                bw = ((x2 - x1) / W).clamp(0, 1)
                bh = ((y2 - y1) / H).clamp(0, 1)
                cls = sel[:, 5]
                img_idx = torch.full_like(cls, float(i))
                target_list.append(torch.stack([img_idx, cls, cx, cy, bw, bh], dim=1))
                if target_list is soft_list:
                    conf_list.append(confs[mask])

        dev = images.device
        hard = torch.cat(hard_list, 0).to(dev).float() if hard_list else torch.zeros(0, 6, device=dev)
        soft = torch.cat(soft_list, 0).to(dev).float() if soft_list else torch.zeros(0, 6, device=dev)
        sconf = torch.cat(conf_list, 0).to(dev).float() if conf_list else torch.zeros(0, device=dev)
        return hard, soft, sconf

    def unsupervised_loss(
        self, images: torch.Tensor, teacher: "BaseDetector"
    ) -> Dict[str, torch.Tensor]:
        """3-tier semi-supervised loss (Efficient Teacher PLA).

        Hard pseudo targets (conf >= tau_high): full-weight detection loss.
        Soft pseudo targets (tau_low <= conf < tau_high): loss weighted by
        mean teacher confidence.
        """
        assert isinstance(teacher, YOLOv5Detector), "Teacher must be YOLOv5Detector"
        hard_targets, soft_targets, soft_confs = self._make_pseudo_targets_3tier(
            teacher, images
        )

        n_hard = hard_targets.shape[0]
        n_soft = soft_targets.shape[0]
        self.last_num_pseudo = n_hard + n_soft

        if n_hard == 0 and n_soft == 0:
            return {"det": torch.zeros((), device=images.device)}

        self.yolo.train()
        pred = self.yolo(images)

        loss = torch.zeros((), device=images.device)

        if n_hard > 0:
            loss_hard, _ = self.compute_loss(pred, hard_targets)
            loss = loss + loss_hard

        if n_soft > 0:
            loss_soft, _ = self.compute_loss(pred, soft_targets)
            loss = loss + soft_confs.mean() * loss_soft

        return {"det": loss}
