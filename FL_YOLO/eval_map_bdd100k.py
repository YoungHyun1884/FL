"""BDD100K용 FedSTO 체크포인트를 평가하는 스크립트.

저장된 체크포인트를 불러와 BDD100K validation split에서 YOLOv5 추론을 수행하고,
`ap_per_class`를 이용해 precision / recall / mAP를 계산한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from yolov5.utils.general import non_max_suppression
from yolov5.utils.metrics import ap_per_class

from .yolov5_detector import YOLOv5Detector
from .yolo_dataset_bdd100k import (
    BDD100K_CLASSES,
    NUM_BDD100K_CLASSES,
    make_server_labeled_bdd100k,
)


DATA_ROOT = Path("/home/pyh/바탕화면/FL/dataset/BDD100k")

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    # 일부 YOLOv5 버전은 아직 np.trapz를 호출한다.
    np.trapz = np.trapezoid


def load_model(ckpt_path: str, device: str) -> YOLOv5Detector:
    """FedSTO 체크포인트에서 YOLOv5Detector를 불러온다."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = YOLOv5Detector(num_classes=NUM_BDD100K_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def val_collate(batch):
    """이미지와 이미지별 target 텐서를 묶어 반환한다."""
    imgs = torch.stack([b["images"] for b in batch])
    targets_list = [b["targets"] for b in batch]
    return imgs, targets_list


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """두 박스 집합의 IoU를 계산한다. (N, 4) x (M, 4) -> (N, M)"""
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


@torch.no_grad()
def run_eval(
    model: YOLOv5Detector,
    val_ds,
    device: str,
    img_size: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    batch_size: int = 16,
    num_workers: int = 0,
):
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collate,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
    )

    all_stats = []  # (correct, conf, pred_cls, target_cls)
    num_targets_per_class = torch.zeros(NUM_BDD100K_CLASSES)
    iouv = torch.linspace(0.5, 0.95, 10, device=device)

    n_batches = len(loader)
    for bi, (imgs, targets_list) in enumerate(loader):
        if bi % 20 == 0:
            print(f"  eval batch {bi}/{n_batches} ...", flush=True)

        imgs = imgs.to(device, non_blocking=True)
        out = model.yolo(imgs)
        preds = out[0] if isinstance(out, (tuple, list)) else out
        nms_preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)

        for i, pred in enumerate(nms_preds):
            gt = targets_list[i]  # (N, 5) [cls, cx, cy, w, h] 정규화 좌표
            n_gt = gt.shape[0]

            if n_gt > 0:
                for c in gt[:, 0].int():
                    num_targets_per_class[c] += 1

            if pred is None or len(pred) == 0:
                if n_gt > 0:
                    all_stats.append(
                        (
                            torch.zeros(0, len(iouv), dtype=torch.bool, device=device),
                            torch.zeros(0, device=device),
                            torch.zeros(0, device=device),
                            gt[:, 0].to(device),
                        )
                    )
                continue

            pred_boxes = pred[:, :4]
            pred_conf = pred[:, 4]
            pred_cls = pred[:, 5]

            if n_gt == 0:
                all_stats.append(
                    (
                        torch.zeros(len(pred), len(iouv), dtype=torch.bool, device=device),
                        pred_conf,
                        pred_cls,
                        torch.zeros(0, device=device),
                    )
                )
                continue

            gt_cls = gt[:, 0].to(device)
            gt_boxes = torch.zeros(n_gt, 4, device=device)
            gt_boxes[:, 0] = (gt[:, 1] - gt[:, 3] / 2) * img_size
            gt_boxes[:, 1] = (gt[:, 2] - gt[:, 4] / 2) * img_size
            gt_boxes[:, 2] = (gt[:, 1] + gt[:, 3] / 2) * img_size
            gt_boxes[:, 3] = (gt[:, 2] + gt[:, 4] / 2) * img_size

            iou = box_iou(pred_boxes, gt_boxes)
            cls_match = pred_cls[:, None] == gt_cls[None, :]
            iou_masked = iou * cls_match.float()

            correct = torch.zeros(len(pred), len(iouv), dtype=torch.bool, device=device)
            for j, iou_thresh in enumerate(iouv):
                matches = (iou_masked >= iou_thresh).nonzero(as_tuple=False)
                if matches.numel() == 0:
                    continue
                iou_vals = iou_masked[matches[:, 0], matches[:, 1]]
                order = iou_vals.argsort(descending=True)
                matches = matches[order]
                pred_matched = set()
                gt_matched = set()
                for pi, gi in matches.tolist():
                    if pi in pred_matched or gi in gt_matched:
                        continue
                    correct[pi, j] = True
                    pred_matched.add(pi)
                    gt_matched.add(gi)

            all_stats.append((correct, pred_conf, pred_cls, gt_cls))

    if not all_stats:
        raise RuntimeError("No predictions or targets were collected during evaluation.")

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*all_stats)]
    if len(stats) == 0 or len(stats[0]) == 0:
        raise RuntimeError("No detections found during evaluation.")

    tp, conf, pred_cls, target_cls = stats
    names = {i: name for i, name in enumerate(BDD100K_CLASSES)}
    tp_curve, fp_curve, p, r, f1, ap, classes = ap_per_class(
        tp, conf, pred_cls, target_cls, names=names
    )

    ap50 = ap[:, 0] if ap.ndim == 2 else ap
    ap5095 = ap.mean(1) if ap.ndim == 2 else ap

    print(
        f"\n{'Class':<15} {'Images':>7} {'Targets':>8} {'P':>7} "
        f"{'R':>7} {'mAP@.5':>8} {'mAP@.5:.95':>10}"
    )
    print("-" * 70)

    for i, c in enumerate(classes.astype(int)):
        name = BDD100K_CLASSES[c] if c < len(BDD100K_CLASSES) else f"cls_{c}"
        nt = int(num_targets_per_class[c])
        print(
            f"{name:<15} {len(val_ds):>7} {nt:>8} {p[i]:>7.3f} {r[i]:>7.3f} "
            f"{ap50[i]:>8.3f} {ap5095[i]:>10.3f}"
        )

    mp, mr = p.mean(), r.mean()
    map50 = ap50.mean()
    map5095 = ap5095.mean()
    print("-" * 70)
    print(
        f"{'all':<15} {len(val_ds):>7} {int(num_targets_per_class.sum()):>8} "
        f"{mp:>7.3f} {mr:>7.3f} {map50:>8.3f} {map5095:>10.3f}"
    )

    return {
        "precision": float(mp),
        "recall": float(mr),
        "map50": float(map50),
        "map5095": float(map5095),
    }


# 논문 표 1/2/3에 맞춘 4개 날씨 그룹 평가
WEATHER_GROUPS = {
    "Cloudy": {"weather": ["clear", "partly cloudy"], "timeofday": None},
    "Overcast": {"weather": ["overcast"], "timeofday": None},
    "Rainy": {"weather": ["rainy"], "timeofday": None},
    "Snowy": {"weather": ["snowy"], "timeofday": None},
}


def _split_val_json_by_weather(ann_file: str):
    """val JSON 항목을 속성 기준으로 날씨 그룹별로 나눈다."""
    import json as _json
    with open(ann_file, "r") as f:
        all_entries = _json.load(f)

    groups = {name: [] for name in WEATHER_GROUPS}
    for entry in all_entries:
        attrs = entry.get("attributes", {})
        w = attrs.get("weather", "unknown")
        t = attrs.get("timeofday", "unknown")
        for group_name, criteria in WEATHER_GROUPS.items():
            weather_match = w in criteria["weather"]
            time_match = criteria["timeofday"] is None or t in criteria["timeofday"]
            if weather_match and time_match:
                groups[group_name].append(entry)
                break  # 각 이미지는 하나의 그룹에만 속한다.
    return groups


def run_per_weather_eval(
    model: YOLOv5Detector,
    data_root: str,
    device: str,
    img_size: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """논문처럼 날씨 그룹별로 성능을 평가한다.

    train/val 데이터 누수를 피하기 위해 날씨 속성이 포함된 val JSON을 사용한다.
    """
    root = Path(data_root)
    img_dir = root / "images" / "100k" / "val"
    ann_file = root / "labels" / "bdd100k_labels_images_val.json"

    from .yolo_dataset_bdd100k import BDD100KJsonLabeled, _BDD_CAT_TO_IDX

    weather_groups = _split_val_json_by_weather(str(ann_file))
    all_metrics = {}

    for weather_name, entries in weather_groups.items():
        if not entries:
            print(f"\n[{weather_name}] No data found, skipping.")
            continue

        # 필터링된 항목으로 데이터셋을 만든다.
        ds = BDD100KJsonLabeled(
            img_dir=str(img_dir),
            ann_file=str(ann_file),
            img_size=img_size,
            augment=False,
        )
        # 날씨로 걸러진 항목만 남기도록 entries를 덮어쓴다.
        filtered_entries = []
        for entry in entries:
            labels = entry.get("labels")
            if labels is None:
                continue
            det_labels = [l for l in labels if "box2d" in l and l["category"] in _BDD_CAT_TO_IDX]
            if det_labels:
                filtered_entries.append({"name": entry["name"], "labels": det_labels})
        ds.entries = filtered_entries

        print(f"\n{'='*70}")
        print(f"  Weather: {weather_name} ({len(ds)} images)")
        print(f"{'='*70}")

        metrics = run_eval(
            model, ds, device=device,
            img_size=img_size, conf_thres=conf_thres,
            iou_thres=iou_thres, batch_size=batch_size,
            num_workers=num_workers,
        )
        all_metrics[weather_name] = metrics

    # 요약 표
    print(f"\n{'='*70}")
    print(f"  Per-Weather Summary")
    print(f"{'='*70}")
    print(f"{'Weather':<10} {'Images':>7} {'mAP@.5':>8} {'mAP@.5:.95':>12} {'P':>7} {'R':>7}")
    print("-" * 55)
    for w, m in all_metrics.items():
        n_imgs = len(weather_groups[w])
        print(f"{w:<10} {n_imgs:>7} {m['map50']:>8.3f} {m['map5095']:>12.3f} {m['precision']:>7.3f} {m['recall']:>7.3f}")
    if all_metrics:
        avg_map50 = sum(m["map50"] for m in all_metrics.values()) / len(all_metrics)
        avg_map5095 = sum(m["map5095"] for m in all_metrics.values()) / len(all_metrics)
        print("-" * 55)
        print(f"{'Average':<10} {'':>7} {avg_map50:>8.3f} {avg_map5095:>12.3f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="BDD100K mAP evaluation for FedSTO")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints_bdd100k/global_phase2.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DATA_ROOT),
        help="Path to BDD100K root directory",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on val images for a quick smoke evaluation",
    )
    parser.add_argument(
        "--per-weather",
        action="store_true",
        help="Evaluate per weather condition (Clear/Night/Rainy/Snowy) like the paper",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    model = load_model(args.ckpt, device)
    print("Model loaded.")

    if args.per_weather:
        run_per_weather_eval(
            model,
            data_root=args.data_root,
            device=device,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        val_ds = make_server_labeled_bdd100k(
            data_root=args.data_root,
            split="val",
            img_size=args.img_size,
            max_images=args.max_images,
            augment=False,
        )
        print(f"Val dataset: {len(val_ds)} images, {NUM_BDD100K_CLASSES} classes")

        metrics = run_eval(
            model,
            val_ds,
            device=device,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"\nSummary: {metrics}")


if __name__ == "__main__":
    main()
