"""SSLAD2D 전용 평가 파일
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict일

import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

from yolov5.utils.general import non_max_suppression
from yolov5.utils.metrics import ap_per_class

from .yolov5_detector import YOLOv5Detector
from .yolo_dataset_SSLAD2D import SSLAD2DDataset, SSLAD_CLASSES, NUM_SSLAD_CLASSES


# ---- 경로 설정 ----
DATA_ROOT = Path("/home/pyh/바탕화면/FL/dataset")
LABELED_ROOT = DATA_ROOT / "labeled" / "labeled_trainval" / "SSLAD-2D" / "labeled"
VAL_IMG_DIR = LABELED_ROOT / "val"
VAL_ANN = LABELED_ROOT / "annotations" / "instance_val.json"


def load_model(ckpt_path: str, device: str) -> YOLOv5Detector:
    """FedSTO 체크포인트에서 YOLOv5Detector를 불러온다."""
    model = YOLOv5Detector(num_classes=NUM_SSLAD_CLASSES)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def val_collate(batch):
    """이미지, 이미지별 target 목록, 원본 크기를 함께 반환하는 collate 함수."""
    imgs = torch.stack([b["images"] for b in batch])
    targets_list = [b["targets"] for b in batch]
    orig_sizes = [b["orig_size"] for b in batch]
    return imgs, targets_list, orig_sizes


class ValDataset(SSLAD2DDataset):
    """원본 이미지 크기까지 함께 반환하도록 확장한 validation 데이터셋."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        img_info = self.images[idx]
        item["orig_size"] = (img_info["width"], img_info["height"])
        return item


@torch.no_grad()
def run_eval(
    model: YOLOv5Detector,
    val_ds: ValDataset,
    device: str,
    img_size: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    batch_size: int = 16,
    num_workers: int = 4,
):
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=val_collate, num_workers=num_workers, pin_memory=True,
    )

    all_stats = []  # (correct, conf, pred_cls, target_cls) 목록
    num_targets_per_class = torch.zeros(NUM_SSLAD_CLASSES)

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # mAP@0.5:0.95용 IoU 임계값

    n_batches = len(loader)
    for bi, (imgs, targets_list, orig_sizes) in enumerate(loader):
        if bi % 20 == 0:
            print(f"  eval batch {bi}/{n_batches} ...")
        imgs = imgs.to(device)

        # 추론
        out = model.yolo(imgs)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        nms_preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)

        # 이미지별 평가
        for i, pred in enumerate(nms_preds):
            gt = targets_list[i]  # (N, 5) [cls, cx, cy, w, h] 정규화 좌표
            n_gt = gt.shape[0]

            if n_gt > 0:
                for c in gt[:, 0].int():
                    num_targets_per_class[c] += 1

            if pred is None or len(pred) == 0:
                if n_gt > 0:
                    all_stats.append((
                        torch.zeros(0, len(iouv), dtype=torch.bool, device=device),
                        torch.zeros(0, device=device),
                        torch.zeros(0, device=device),
                        gt[:, 0].to(device),
                    ))
                continue

            # pred: (K, 6) [x1, y1, x2, y2, conf, cls] 픽셀 좌표
            pred_boxes = pred[:, :4]
            pred_conf = pred[:, 4]
            pred_cls = pred[:, 5]

            if n_gt == 0:
                all_stats.append((
                    torch.zeros(len(pred), len(iouv), dtype=torch.bool, device=device),
                    pred_conf,
                    pred_cls,
                    torch.zeros(0, device=device),
                ))
                continue

            # GT를 정규화 좌표에서 픽셀 좌표로 변환한다.
            gt_cls = gt[:, 0].to(device)
            gt_boxes = torch.zeros(n_gt, 4, device=device)
            gt_boxes[:, 0] = (gt[:, 1] - gt[:, 3] / 2) * img_size
            gt_boxes[:, 1] = (gt[:, 2] - gt[:, 4] / 2) * img_size
            gt_boxes[:, 2] = (gt[:, 1] + gt[:, 3] / 2) * img_size
            gt_boxes[:, 3] = (gt[:, 2] + gt[:, 4] / 2) * img_size

            # 예측과 정답 사이 IoU를 계산한다.
            iou = box_iou(pred_boxes, gt_boxes)

            # 클래스가 같은 경우만 매칭 후보로 남긴다.
            cls_match = pred_cls[:, None] == gt_cls[None, :]  # (K, M)
            iou_masked = iou * cls_match.float()

            correct = torch.zeros(len(pred), len(iouv), dtype=torch.bool, device=device)
            for j, iou_thresh in enumerate(iouv):
                # IoU가 큰 순서대로 greedy matching을 수행한다.
                matches = (iou_masked >= iou_thresh).nonzero(as_tuple=False)
                if matches.numel():
                    iou_vals = iou_masked[matches[:, 0], matches[:, 1]]
                    order = iou_vals.argsort(descending=True)
                    matches = matches[order]
                    pred_matched = set()
                    gt_matched_set = set()
                    for pi, gi in matches.tolist():
                        if pi in pred_matched or gi in gt_matched_set:
                            continue
                        correct[pi, j] = True
                        pred_matched.add(pi)
                        gt_matched_set.add(gi)

            all_stats.append((correct, pred_conf, pred_cls, gt_cls))

    # 통계를 하나로 합친다.
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*all_stats)]

    if len(stats) == 0 or len(stats[0]) == 0:
        print("No detections found!")
        return

    # yolov5의 ap_per_class로 AP를 계산한다.
    tp, conf, pred_cls, target_cls = stats
    names = {i: name for i, name in enumerate(SSLAD_CLASSES)}
    results = ap_per_class(tp, conf, pred_cls, target_cls, names=names)

    # ap_per_class 반환값: (tp, fp, p, r, f1, ap, unique_classes)
    tp_curve, fp_curve, p, r, f1, ap, classes = results

    # ap 모양: (num_classes, num_iou_thresholds)
    ap50 = ap[:, 0] if ap.ndim == 2 else ap
    ap5095 = ap.mean(1) if ap.ndim == 2 else ap

    # 결과 출력
    print(f"\n{'Class':<15} {'Images':>7} {'Targets':>8} {'P':>7} {'R':>7} {'mAP@.5':>8} {'mAP@.5:.95':>10}")
    print("-" * 70)

    for i, c in enumerate(classes.astype(int)):
        name = SSLAD_CLASSES[c] if c < len(SSLAD_CLASSES) else f"cls_{c}"
        nt = int(num_targets_per_class[c])
        print(f"{name:<15} {len(val_ds):>7} {nt:>8} {p[i]:>7.3f} {r[i]:>7.3f} {ap50[i]:>8.3f} {ap5095[i]:>10.3f}")

    # 전체 결과
    mp, mr = p.mean(), r.mean()
    map50 = ap50.mean()
    map5095 = ap5095.mean()
    print("-" * 70)
    print(f"{'all':<15} {len(val_ds):>7} {int(num_targets_per_class.sum()):>8} {mp:>7.3f} {mr:>7.3f} {map50:>8.3f} {map5095:>10.3f}")
    print()


def box_iou(box1, box2):
    """두 박스 집합의 IoU를 계산한다. (N, 4) x (M, 4) -> (N, M)"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-7)


def main():
    parser = argparse.ArgumentParser(description="mAP evaluation for FedSTO")
    parser.add_argument("--ckpt", type=str, default="./checkpoints_sslad/global_phase2.pt",
                        help="Path to checkpoint")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-thres", type=float, default=0.01)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    model = load_model(args.ckpt, device)
    print("Model loaded.")

    val_ds = ValDataset(
        img_dir=str(VAL_IMG_DIR),
        ann_file=str(VAL_ANN),
        img_size=args.img_size,
    )
    print(f"Val dataset: {len(val_ds)} images, {NUM_SSLAD_CLASSES} classes")

    run_eval(
        model, val_ds, device,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
