"""BDD100K 체크포인트의 pseudo-label 품질을 점검하는 스크립트.

체크포인트의 pseudo-label 생성 과정을 labeled validation set에서 확인한다.
클라이언트의 실제 unlabeled 품질을 직접 재는 것은 아니지만, 같은 teacher/NMS/
threshold 로직을 정답과 비교해 대략적인 품질을 살펴볼 수 있다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from yolov5.utils.general import non_max_suppression

from .yolo_dataset_bdd100k import (
    BDD100K_CLASSES,
    NUM_BDD100K_CLASSES,
    make_server_labeled_bdd100k,
)
from .yolov5_detector import YOLOv5Detector


DATA_ROOT = Path("/home/pyh/바탕화면/FL/dataset/BDD100k")


def load_model(ckpt_path: str, device: str) -> YOLOv5Detector:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = YOLOv5Detector(num_classes=NUM_BDD100K_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def val_collate(batch):
    imgs = torch.stack([b["images"] for b in batch])
    targets_list = [b["targets"] for b in batch]
    return imgs, targets_list


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def yolo_to_xyxy(targets: torch.Tensor, img_size: int, device: str):
    if targets.numel() == 0:
        return (
            torch.zeros(0, 4, device=device),
            torch.zeros(0, device=device),
        )
    gt = targets.to(device)
    gt_cls = gt[:, 0]
    gt_boxes = torch.zeros(gt.shape[0], 4, device=device)
    gt_boxes[:, 0] = (gt[:, 1] - gt[:, 3] / 2) * img_size
    gt_boxes[:, 1] = (gt[:, 2] - gt[:, 4] / 2) * img_size
    gt_boxes[:, 2] = (gt[:, 1] + gt[:, 3] / 2) * img_size
    gt_boxes[:, 3] = (gt[:, 2] + gt[:, 4] / 2) * img_size
    return gt_boxes, gt_cls


def split_pseudo_by_tier(model: YOLOv5Detector, images: torch.Tensor):
    out = model.yolo(images)
    det = out[0] if isinstance(out, (tuple, list)) else out
    nms_out = non_max_suppression(
        det,
        conf_thres=model.tau_low,
        iou_thres=model.iou_thres,
    )

    per_image = []
    for boxes in nms_out:
        if boxes is None or len(boxes) == 0:
            empty = torch.zeros(0, 4, device=images.device)
            empty1 = torch.zeros(0, device=images.device)
            per_image.append(
                {
                    "hard_boxes": empty,
                    "hard_cls": empty1,
                    "hard_conf": empty1,
                    "soft_boxes": empty,
                    "soft_cls": empty1,
                    "soft_conf": empty1,
                }
            )
            continue

        conf = boxes[:, 4]
        hard_mask = conf >= model.tau_high
        soft_mask = (conf >= model.tau_low) & (conf < model.tau_high)
        per_image.append(
            {
                "hard_boxes": boxes[hard_mask, :4],
                "hard_cls": boxes[hard_mask, 5],
                "hard_conf": boxes[hard_mask, 4],
                "soft_boxes": boxes[soft_mask, :4],
                "soft_cls": boxes[soft_mask, 5],
                "soft_conf": boxes[soft_mask, 4],
            }
        )
    return per_image


def greedy_match(
    pred_boxes: torch.Tensor,
    pred_cls: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    iou_thresh: float,
):
    if pred_boxes.numel() == 0:
        return 0, 0
    if gt_boxes.numel() == 0:
        return 0, 0

    iou = box_iou(pred_boxes, gt_boxes)
    cls_match = pred_cls[:, None] == gt_cls[None, :]
    iou_masked = iou * cls_match.float()
    matches = (iou_masked >= iou_thresh).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return 0, 0

    iou_vals = iou_masked[matches[:, 0], matches[:, 1]]
    matches = matches[iou_vals.argsort(descending=True)]

    matched_pred = set()
    matched_gt = set()
    for pi, gi in matches.tolist():
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
    return len(matched_pred), len(matched_gt)


@torch.no_grad()
def inspect_pseudo_labels(
    model: YOLOv5Detector,
    val_ds,
    device: str,
    img_size: int = 640,
    iou_thresh: float = 0.5,
    batch_size: int = 4,
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

    stats = {
        "images": 0,
        "gt_total": 0,
        "hard_total": 0,
        "soft_total": 0,
        "hard_tp": 0,
        "soft_tp": 0,
        "all_tp": 0,
        "hard_gt_hit": 0,
        "soft_gt_hit": 0,
        "all_gt_hit": 0,
        "soft_conf_sum": 0.0,
        "soft_conf_count": 0,
    }
    cls_pred_counts = torch.zeros(NUM_BDD100K_CLASSES, dtype=torch.long)
    cls_gt_counts = torch.zeros(NUM_BDD100K_CLASSES, dtype=torch.long)

    n_batches = len(loader)
    for bi, (imgs, targets_list) in enumerate(loader):
        if bi % 20 == 0:
            print(f"  inspect batch {bi}/{n_batches} ...", flush=True)

        imgs = imgs.to(device, non_blocking=True)
        pseudo = split_pseudo_by_tier(model, imgs)

        for i, p in enumerate(pseudo):
            gt_boxes, gt_cls = yolo_to_xyxy(targets_list[i], img_size=img_size, device=device)
            hard_boxes, hard_cls = p["hard_boxes"], p["hard_cls"]
            soft_boxes, soft_cls = p["soft_boxes"], p["soft_cls"]
            all_boxes = torch.cat([hard_boxes, soft_boxes], dim=0)
            all_cls = torch.cat([hard_cls, soft_cls], dim=0)

            n_gt = gt_boxes.shape[0]
            n_hard = hard_boxes.shape[0]
            n_soft = soft_boxes.shape[0]

            hard_tp, hard_gt_hit = greedy_match(hard_boxes, hard_cls, gt_boxes, gt_cls, iou_thresh)
            soft_tp, soft_gt_hit = greedy_match(soft_boxes, soft_cls, gt_boxes, gt_cls, iou_thresh)
            all_tp, all_gt_hit = greedy_match(all_boxes, all_cls, gt_boxes, gt_cls, iou_thresh)

            stats["images"] += 1
            stats["gt_total"] += n_gt
            stats["hard_total"] += n_hard
            stats["soft_total"] += n_soft
            stats["hard_tp"] += hard_tp
            stats["soft_tp"] += soft_tp
            stats["all_tp"] += all_tp
            stats["hard_gt_hit"] += hard_gt_hit
            stats["soft_gt_hit"] += soft_gt_hit
            stats["all_gt_hit"] += all_gt_hit
            stats["soft_conf_sum"] += float(p["soft_conf"].sum().item())
            stats["soft_conf_count"] += int(p["soft_conf"].numel())

            for c in gt_cls.int().cpu():
                cls_gt_counts[c] += 1
            for c in all_cls.int().cpu():
                if 0 <= int(c) < NUM_BDD100K_CLASSES:
                    cls_pred_counts[int(c)] += 1

    def safe_div(a, b):
        return float(a) / float(b) if b else 0.0

    summary = {
        "images": stats["images"],
        "avg_gt_per_image": safe_div(stats["gt_total"], stats["images"]),
        "avg_hard_per_image": safe_div(stats["hard_total"], stats["images"]),
        "avg_soft_per_image": safe_div(stats["soft_total"], stats["images"]),
        "hard_precision@iou": safe_div(stats["hard_tp"], stats["hard_total"]),
        "soft_precision@iou": safe_div(stats["soft_tp"], stats["soft_total"]),
        "all_precision@iou": safe_div(stats["all_tp"], stats["hard_total"] + stats["soft_total"]),
        "hard_recall@iou": safe_div(stats["hard_gt_hit"], stats["gt_total"]),
        "soft_recall@iou": safe_div(stats["soft_gt_hit"], stats["gt_total"]),
        "all_recall@iou": safe_div(stats["all_gt_hit"], stats["gt_total"]),
        "mean_soft_conf": safe_div(stats["soft_conf_sum"], stats["soft_conf_count"]),
    }

    print("\nPseudo-label inspection summary")
    print("-" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:<24} {v:.6f}")
        else:
            print(f"{k:<24} {v}")

    print("\nClass distribution: GT vs pseudo")
    print("-" * 60)
    print(f"{'Class':<15} {'GT':>10} {'Pseudo':>10}")
    for i, name in enumerate(BDD100K_CLASSES):
        gt_n = int(cls_gt_counts[i].item())
        pred_n = int(cls_pred_counts[i].item())
        if gt_n == 0 and pred_n == 0:
            continue
        print(f"{name:<15} {gt_n:>10} {pred_n:>10}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect pseudo-label quality on BDD100K val")
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    model = load_model(args.ckpt, device)
    print(
        f"Model loaded. tau_low={model.tau_low:.3f}, "
        f"tau_high={model.tau_high:.3f}, iou_thres={model.iou_thres:.3f}"
    )

    val_ds = make_server_labeled_bdd100k(
        data_root=args.data_root,
        split="val",
        img_size=args.img_size,
        max_images=args.max_images,
        augment=False,
    )
    print(f"Val dataset: {len(val_ds)} images")

    inspect_pseudo_labels(
        model,
        val_ds,
        device=device,
        img_size=args.img_size,
        iou_thresh=args.iou_thresh,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
