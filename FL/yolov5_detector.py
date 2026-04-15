"""YOLOv5 adapter implementing BaseDetector.
  - supervised path:
      labeled image + GT target -> YOLO forward -> ComputeLoss
  - unsupervised path:
      unlabeled image -> EMA teacher inference -> NMS -> pseudo target 생성
      -> student YOLO forward -> pseudo target 기준 ComputeLoss
  - Phase 1에서는 backbone만 서버에 집계되고,
    Phase 2에서는 전체 파라미터가 집계된다.

F
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

# 모델 크기 문자와 COCO 사전학습 가중치 파일명을 매핑한다.
_COCO_WEIGHT_NAMES = {
    "n": "yolov5n.pt",
    "s": "yolov5s.pt",
    "m": "yolov5m.pt",
    "l": "yolov5l.pt",
    "x": "yolov5x.pt",
}


def default_hyp() -> dict:
    """ComputeLoss에 필요한 최소 YOLOv5 hyp 딕셔너리다. 값은 FedSTO 논문을 따른다."""
    return {
        "box": 0.05,
        "cls": 0.3,           # 논문 기준 cls loss balance = 0.3
        "cls_pw": 1.0,
        "obj": 0.7,           # 논문 기준 obj loss balance = 0.7
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
    """yaml 파일명에서 모델 크기 문자(n/s/m/l/x)를 추출한다."""
    basename = os.path.basename(yaml_path)          # 예: "yolov5s.yaml"
    for sz in ("n", "s", "m", "l", "x"):
        if f"yolov5{sz}" in basename:
            return sz
    return "l"  # 기본값


def _ensure_yolov5_module_aliases() -> None:
    """yolov5.models/yolov5.utils를 짧은 이름으로도 등록한다.

    Ultralytics COCO 체크포인트는 ``models.yolo.Model`` 같은 경로를 pickle에 기록한다.
    unpickle할 때 Python이 ``import models``를 해석할 수 있어야 하므로,
    pip로 설치된 ``yolov5.models`` / ``yolov5.utils``에 대해
    ``sys.modules`` 별칭을 한 번 만들어 둔다.
    """
    import sys
    import yolov5

    for sub in ("models", "utils"):
        full = f"yolov5.{sub}"
        if full in sys.modules and sub not in sys.modules:
            sys.modules[sub] = sys.modules[full]

    # 체크포인트가 참조할 수 있는 하위 모듈에도 같은 방식으로 별칭을 건다.
    for key, mod in list(sys.modules.items()):
        if key.startswith("yolov5.models.") or key.startswith("yolov5.utils."):
            short = key[len("yolov5."):]        # 예: "models.yolo"
            if short not in sys.modules:
                sys.modules[short] = mod


def _load_coco_pretrained(model: YoloModel, size: str) -> None:
    """COCO 사전학습 가중치를 내려받아 모양이 맞는 파라미터만 옮긴다.

    num_classes != 80 등으로 모양이 다른 detection head 층은 건너뛰어
    어떤 클래스 수에서도 모델을 사용할 수 있게 한다.
    """
    _ensure_yolov5_module_aliases()

    weight_file = _COCO_WEIGHT_NAMES.get(size, "yolov5l.pt")
    local_path = attempt_download(weight_file)
    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    # ultralytics 체크포인트는 모델을 'ema' 또는 'model' 아래에 저장한다.
    # 'ema'가 None이 아니면 우선 사용하고, 아니면 'model'을 사용한다.
    coco_model = ckpt.get("ema") or ckpt.get("model")
    if coco_model is None:
        logger.warning("Could not find model in checkpoint %s — skipping pretrained load", weight_file)
        return

    if hasattr(coco_model, "state_dict"):
        coco_sd = coco_model.float().state_dict()
    else:
        coco_sd = coco_model

    # 현재 실험용 detector의 state_dict와 COCO checkpoint를 키/shape 기준으로 맞춘다.
    # 클래스 수가 다르면 마지막 detect head 일부는 자연스럽게 skip된다.
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
    """YOLOv5를 FedSTO에 맞춰 감싼 래퍼다.

    Args:
        num_classes: 객체 클래스 수
        yaml_path: yolov5{n,s,m,l,x}.yaml 경로. 기본값은 yolov5l
        split_idx: backbone과 non-backbone을 나누는 레이어 인덱스 경계
                   기본값 10은 YOLOv5 v6.0 기준과 맞는다.
        tau_high: hard pseudo-label objectness/score 임계값
        tau_low:  NMS confidence 임계값 (이보다 낮은 박스는 고려하지 않음)
        iou_thres: NMS IoU 임계값
        hyp: YOLOv5 손실 하이퍼파라미터. 기본값은 `default_hyp()` 사용
    """

    # 아래 `_is_backbone` / `_is_non_backbone`를 재정의해 실제 분할을 수행하므로
    # 여기 값들은 자리만 잡아둔 것이다. (YOLOv5는 평평한 layer index를 사용한다.)
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
        # 실제 YOLOv5 본체다. FedSTO는 이 모듈을 감싸서 FL용 인터페이스만 제공한다.
        self.yolo = YoloModel(yaml_path, ch=3, nc=num_classes)

        if pretrained:
            size = _infer_size_from_yaml(yaml_path)
            _load_coco_pretrained(self.yolo, size)

        self.yolo.hyp = hyp or default_hyp()
        self.yolo.gr = 1.0  # gr = 1.0이면 obj loss에 IoU 비율을 전부 반영한다.
        # ComputeLoss는 YOLO 내부 anchor/head 설정을 읽어 손실 계산기를 만든다.
        self.compute_loss = ComputeLoss(self.yolo)

    # ------- device 이동 --------------------------------------------
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # ComputeLoss는 생성 시점의 self.device(CPU)를 내부에 캐시한다.
        # model.to("cuda") 이후에는 anchor와 device가 맞도록 다시 만든다.
        result.compute_loss = ComputeLoss(result.yolo)
        return result

    # ------- forward ----------------------------------------------------
    def forward(self, x: torch.Tensor):
        return self.yolo(x)

    # ------- 파라미터 분할 (기반 클래스 재정의) --------------
    def _layer_idx_from_name(self, name: str) -> int | None:
        if not name.startswith(self._PARAM_PREFIX):
            return None
        rest = name[len(self._PARAM_PREFIX):]
        head = rest.split(".", 1)[0]
        return int(head) if head.isdigit() else None

    def _is_backbone(self, name: str) -> bool:
        # YOLOv5는 backbone/neck/head가 하나의 평평한 layer sequence라서
        # 파라미터 이름에서 layer index를 뽑아 split_idx 이전이면 backbone으로 본다.
        idx = self._layer_idx_from_name(name)
        return idx is not None and idx < self.split_idx

    def _is_non_backbone(self, name: str) -> bool:
        # split_idx 이후는 PANet neck + detect head에 해당한다.
        idx = self._layer_idx_from_name(name)
        return idx is not None and idx >= self.split_idx

    def non_backbone_weight_matrices(self) -> Iterable[torch.Tensor]:
        """직교 정규화용으로 neck + head의 Conv/Linear 가중치를 반환한다."""
        seq = self.yolo.model  # 0..N 레이어로 이뤄진 nn.Sequential
        for layer in seq[self.split_idx:]:
            for m in layer.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    yield m.weight

    # ------- 손실 함수 -----------------------------------------------------
    def supervised_loss(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """YOLOv5 지도 손실을 계산한다.

        Args:
            images: (B, 3, H, W)
            targets: 정규화된 (N, 6) [img_idx, cls, cx, cy, w, h]
        """
        # labeled batch는 그대로 YOLO의 raw multi-scale output과 ComputeLoss로 연결된다.
        self.yolo.train()
        pred = self.yolo(images)  # train 모드에서는 길이 3의 텐서 리스트가 나온다.
        loss, _loss_items = self.compute_loss(pred, targets)
        # 프레임워크의 sum()이 원래 스케일을 유지하도록 항목 하나만 반환한다.
        return {"det": loss}

    @torch.no_grad()
    def _make_pseudo_targets_3tier(
        self, teacher: "YOLOv5Detector", images: torch.Tensor
    ) -> tuple:
        """클라이언트가 선생이 되어 문제집을 만들고, 문제는 신뢰도 기준으로 hard/soft로 나뉜다.
        학생은 그 문제를 풀면서 학습한다.
        모든 예측을 그대로 믿지 않고 신뢰도에 따라 손실 가중치를 조절해
        잘못된 정보로부터 모델이 망가지는 것을 막는다.
        """
        # 1) EMA teacher는 gradient 없이 비라벨 이미지를 추론한다.
        teacher.yolo.eval()
        out = teacher.yolo(images) # 정답이 없는 이미지를 선생에게 넣어 예측값을 얻는다.
        det = out[0] if isinstance(out, (tuple, list)) else out  # (B, N, 5+nc)
        # 2) NMS로 중복된 박스를 정리하고, 너무 낮은 점수는 tau_low 아래에서 버린다.
        nms_out = non_max_suppression(
            det, conf_thres=self.tau_low, iou_thres=self.iou_thres
        )

        H, W = images.shape[-2:]
        hard_list, soft_list, conf_list = [], [], []
        for i, boxes in enumerate(nms_out):
            if boxes is None or len(boxes) == 0:
                continue
            # boxes: (K, 6) [x1, y1, x2, y2, conf, cls]
            confs = boxes[:, 4]

            for mask, target_list in [ # 선생이 예측한 박스를 hard / soft으로 나누는 기준은 confidence score가 tau_high 이상인지, tau_low 이상 tau_high 미만인지이다.
                (confs >= self.tau_high, hard_list),
                ((confs >= self.tau_low) & (confs < self.tau_high), soft_list),
            ]:
                sel = boxes[mask]
                if len(sel) == 0:
                    continue
                # 3) teacher가 뽑은 픽셀 bbox를 다시 YOLO 손실용 정규화 타깃으로 바꾼다.
                x1, y1, x2, y2 = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
                cx = ((x1 + x2) * 0.5 / W).clamp(0, 1)
                cy = ((y1 + y2) * 0.5 / H).clamp(0, 1)
                bw = ((x2 - x1) / W).clamp(0, 1)
                bh = ((y2 - y1) / H).clamp(0, 1)
                cls = sel[:, 5]
                img_idx = torch.full_like(cls, float(i))
                target_list.append(torch.stack([img_idx, cls, cx, cy, bw, bh], dim=1))
                if target_list is soft_list:
                    # soft target은 나중에 confidence 평균으로 손실을 약하게 건다.
                    conf_list.append(confs[mask])

        dev = images.device
        hard = torch.cat(hard_list, 0).to(dev).float() if hard_list else torch.zeros(0, 6, device=dev)
        soft = torch.cat(soft_list, 0).to(dev).float() if soft_list else torch.zeros(0, 6, device=dev)
        sconf = torch.cat(conf_list, 0).to(dev).float() if conf_list else torch.zeros(0, device=dev)
        return hard, soft, sconf

    def unsupervised_loss( # 만든 가짜 정답을 가지고 학생 모델이 실제로 공부하게 만드는 함수
        self, images: torch.Tensor, teacher: "BaseDetector"
    ) -> Dict[str, torch.Tensor]:
        
        assert isinstance(teacher, YOLOv5Detector), "Teacher must be YOLOv5Detector"
        # 비라벨 이미지가 이 함수 안에서 가짜 라벨이 붙은 배치로 변환된다.
        hard_targets, soft_targets, soft_confs = self._make_pseudo_targets_3tier(
            teacher, images
        )

        n_hard = hard_targets.shape[0]
        n_soft = soft_targets.shape[0]
        self.last_num_pseudo = n_hard + n_soft

        if n_hard == 0 and n_soft == 0:
            # teacher가 쓸 만한 박스를 하나도 못 찾으면 이 배치는 건너뛴다.
            return {"det": torch.zeros((), device=images.device)}

        # student는 같은 이미지를 다시 forward해서 가짜 타깃과 맞는 방향으로 학습한다.
        self.yolo.train()
        pred = self.yolo(images)

        loss = torch.zeros((), device=images.device)

        if n_hard > 0: 
            # high-confidence pseudo는 실제 GT처럼 강하게 사용한다.
            loss_hard, _ = self.compute_loss(pred, hard_targets)
            loss = loss + loss_hard

        if n_soft > 0:
            # low/high 사이의 pseudo는 teacher 평균 confidence만큼만 반영한다.
            loss_soft, _ = self.compute_loss(pred, soft_targets)
            loss = loss + soft_confs.mean() * loss_soft

        return {"det": loss}
