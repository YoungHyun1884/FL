"""YOLOv5를 FedSTO 규칙에 맞게 연결
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

# 모델 크기 문자와 COCO 사전학습 가중치 파일 이름 매핑
_COCO_WEIGHT_NAMES = {
    "n": "yolov5n.pt",
    "s": "yolov5s.pt",
    "m": "yolov5m.pt",
    "l": "yolov5l.pt",
    "x": "yolov5x.pt",
}


def default_hyp() -> dict:
    """ComputeLoss에 필요한 최소 YOLOv5 hyp 설정을 반환한다."""
    return {
        "box": 0.05,
        "cls": 0.3,           # 논문에서 사용한 cls loss 비율
        "cls_pw": 1.0,
        "obj": 0.7,           # 논문에서 사용한 obj loss 비율
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
    """yaml 파일 이름에서 모델 크기 문자(n/s/m/l/x)를 추출한다."""
    basename = os.path.basename(yaml_path)
    for sz in ("n", "s", "m", "l", "x"):
        if f"yolov5{sz}" in basename:
            return sz
    return "l"  # 기본값


def _ensure_yolov5_module_aliases() -> None:
    """체크포인트 언피클을 위해 yolov5 모듈 별칭을 등록한다."""
    import sys
    import yolov5

    for sub in ("models", "utils"):
        full = f"yolov5.{sub}"
        if full in sys.modules and sub not in sys.modules:
            sys.modules[sub] = sys.modules[full]

    # 체크포인트가 참조할 수 있는 하위 모듈도 함께 별칭으로 등록한다.
    for key, mod in list(sys.modules.items()):
        if key.startswith("yolov5.models.") or key.startswith("yolov5.utils."):
            short = key[len("yolov5."):]
            if short not in sys.modules:
                sys.modules[short] = mod


def _load_coco_pretrained(model: YoloModel, size: str) -> None:
    """COCO 사전학습 가중치를 내려받아 모양이 맞는 파라미터만 옮긴다."""
    _ensure_yolov5_module_aliases()

    weight_file = _COCO_WEIGHT_NAMES.get(size, "yolov5l.pt")
    local_path = attempt_download(weight_file)
    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    # Ultralytics 체크포인트는 보통 'ema' 또는 'model' 아래에 모델을 저장한다.
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
        self.set_pseudo_config(use_soft_pseudo=True, soft_pseudo_weight=1.0)

        yaml_path = yaml_path or _find_default_yaml("m")
        self.yolo = YoloModel(yaml_path, ch=3, nc=num_classes)

        if pretrained:
            size = _infer_size_from_yaml(yaml_path)
            _load_coco_pretrained(self.yolo, size)

        self.yolo.hyp = hyp or default_hyp()
        self.yolo.gr = 1.0  # obj loss에서 IoU 비율을 전부 반영한다.
        # ComputeLoss는 생성 시점에 anchors / na / nl을 내부에 저장한다.
        self.compute_loss = ComputeLoss(self.yolo)

    # ------- 디바이스 이동 --------------------------------------------
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # ComputeLoss는 초기화 시 device를 저장하므로 모델 이동 후 다시 만든다.
        result.compute_loss = ComputeLoss(result.yolo)
        return result

    # ------- forward ----------------------------------------------------
    def forward(self, x: torch.Tensor):
        return self.yolo(x)

    # ------- 파라미터 분할 --------------------------------
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
        """직교 정규화에 사용할 neck + head의 Conv/Linear 가중치를 반환한다."""
        seq = self.yolo.model  # 레이어 0..N으로 이루어진 nn.Sequential
        for layer in seq[self.split_idx:]:
            for m in layer.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    yield m.weight

    # ------- 손실 함수 --------------------------------
    def supervised_loss( #서버가 labeled 데이터로 학습할 때 쓰는  함수
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """YOLOv5의 supervised loss를 계산한다."""
        self.yolo.train()
        pred = self.yolo(images)  
        loss, _loss_items = self.compute_loss(pred, targets)
        return {"det": loss}

    @torch.no_grad()
    def _make_pseudo_targets_3tier( #teacher EMA 모델이 unlabed 이미지를 보고 수두 박스를 만듦.
        self, teacher: "YOLOv5Detector", images: torch.Tensor
    ) -> tuple:
       
        teacher.yolo.eval() #저기 클라이언트 이미지에 대해 선생모델이 추론. (서버에서 받은 글로벌 weight를 기반으로 로컬 데이터에 대해서 예측해야 하니까)
        out = teacher.yolo(images)
        det = out[0] if isinstance(out, (tuple, list)) else out  
        nms_out = non_max_suppression(
            det, conf_thres=self.tau_low, iou_thres=self.iou_thres
        )
        #수도라벨 생성
        H, W = images.shape[-2:]
        hard_list, soft_list, conf_list = [], [], []
        for i, boxes in enumerate(nms_out):
            if boxes is None or len(boxes) == 0:
                continue
          
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
        
        assert isinstance(teacher, YOLOv5Detector), "Teacher must be YOLOv5Detector"
        hard_targets, soft_targets, soft_confs = self._make_pseudo_targets_3tier(
            teacher, images
        )

        n_hard = hard_targets.shape[0]
        n_soft = soft_targets.shape[0]
        self.last_num_pseudo = n_hard + n_soft

        if n_hard == 0 and n_soft == 0:
            return {"det": torch.zeros((), device=images.device)}

        self.yolo.train() #학생모델이 hard/soft pseudo 타겟에 대해서 예측을 하고 loss를 계산. (학생모델은 백본만 업데이트하니까 글로벌 모델이 학생모델의 백본을 따라가도록 하는 효과)
        pred = self.yolo(images)

        loss = torch.zeros((), device=images.device)

        if n_hard > 0:
            loss_hard, _ = self.compute_loss(pred, hard_targets)
            loss = loss + loss_hard

        if self.use_soft_pseudo and n_soft > 0:
            loss_soft, _ = self.compute_loss(pred, soft_targets)
            loss = loss + (self.soft_pseudo_weight * soft_confs.mean()) * loss_soft

        return {"det": loss}
