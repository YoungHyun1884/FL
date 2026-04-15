"""End-to-end FedSTO run with YOLOv5 on the SSLAD-2D dataset.

Pipeline matches Algorithm 1 of the paper:
    Warmup (server supervised on labeled train)
      -> Phase 1: Selective Training (client backbone only, unlabeled)
      -> Phase 2: FPT + Orthogonal Enhancement

Dataset layout expected under DATA_ROOT:
    labeled/labeled_trainval/SSLAD-2D/labeled/
        annotations/instance_train.json, instance_val.json
        train/*.jpg, val/*.jpg
    labeled/labeled_test/SSLAD-2D/labeled/
        annotations/instance_test.json
        test/*.jpg
    unlabeled/SSLAD-2D/unlabel/image_0/*.jpg
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import FedSTOConfig, OptimConfig
from .orchestrator import FedSTO
from .server import Server
from .client import Client
from .yolov5_detector import YOLOv5Detector
from .yolo_dataset import (
    SSLAD2DDataset,
    NUM_SSLAD_CLASSES,
    labeled_yolo_collate,
    unlabeled_yolo_collate,
    make_non_iid_clients_sslad,
)


# =====================================================================
# Default paths (adjust or pass via CLI)
# =====================================================================
DATA_ROOT = Path("/home/pyh/바탕화면/FL/dataset")

LABELED_ROOT = DATA_ROOT / "labeled" / "labeled_trainval" / "SSLAD-2D" / "labeled"
TRAIN_IMG_DIR = LABELED_ROOT / "train"
TRAIN_ANN = LABELED_ROOT / "annotations" / "instance_train.json"
VAL_IMG_DIR = LABELED_ROOT / "val"
VAL_ANN = LABELED_ROOT / "annotations" / "instance_val.json"

UNLABELED_IMG_DIR = DATA_ROOT / "unlabeled" / "SSLAD-2D" / "unlabel" / "image_0"


@torch.no_grad()
def evaluate(model: YOLOv5Detector, val_ds, device: str, rnd: int, phase: str):
    """Evaluate: mean supervised loss on the held-out labeled val set."""
    loader = DataLoader(val_ds, batch_size=8, collate_fn=labeled_yolo_collate,
                        num_workers=4, pin_memory=True)
    model.eval()
    model.yolo.train()  # ComputeLoss needs train-mode raw outputs
    total, n = 0.0, 0
    for batch in loader:
        imgs = batch["images"].to(device)
        tgts = batch["targets"].to(device)
        loss_dict = model.supervised_loss(imgs, tgts)
        total += float(sum(loss_dict.values()).detach())
        n += 1
        if n >= 50:  # cap for speed
            break
    mean_loss = total / max(n, 1)
    print(f"  [eval@{phase}:r{rnd}] val_det_loss={mean_loss:.4f}")
    return {"val_loss": mean_loss}


def build_yolo(num_classes: int, pretrained: bool = True):
    """Factory for YOLOv5Detector instances (COCO-pretrained by default)."""
    return YOLOv5Detector(
        num_classes=num_classes,
        tau_high=0.5,
        tau_low=0.05,
        iou_thres=0.65,
        pretrained=pretrained,
    )


def main():
    parser = argparse.ArgumentParser(description="FedSTO + YOLOv5 on SSLAD-2D")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for server/client loaders")
    parser.add_argument("--num-clients", type=int, default=5,
                        help="Number of FL clients (split by city)")
    parser.add_argument("--max-per-client", type=int, default=None,
                        help="Cap unlabeled images per client (for dev/debug)")
    parser.add_argument("--warmup-rounds", type=int, default=50)
    parser.add_argument("--t1", type=int, default=100,
                        help="Phase 1 rounds")
    parser.add_argument("--t2", type=int, default=150,
                        help="Phase 2 rounds")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--server-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip COCO pretrained weights (random init)")
    args = parser.parse_args()

    cfg = FedSTOConfig(
        warmup_rounds=args.warmup_rounds,
        T1=args.t1,
        T2=args.t2,
        local_epochs=args.local_epochs,
        server_steps=args.server_steps,
        num_clients=args.num_clients,
        client_sample_ratio=1.0,
        tau_low=0.1, tau_high=0.6,
        tau_low_start=0.1, tau_high_start=0.3,
        use_epoch_adaptor=True,
        unsup_loss_weight=4.0,
        ortho_lambda=1e-3,
        ortho_power_iters=1,
        ema_decay=0.999,
        server_opt=OptimConfig(lr=1e-2, momentum=0.9, weight_decay=5e-4),
        client_opt=OptimConfig(lr=1e-2, momentum=0.9, weight_decay=5e-4),
        device="cuda" if torch.cuda.is_available() else "cpu",
        ckpt_dir="./checkpoints_sslad",
        log_every=1,
    )
    print(f"device = {cfg.device}")

    num_classes = NUM_SSLAD_CLASSES  # 6
    img_size = args.img_size

    # ---- Datasets ----
    print("Loading labeled train dataset...")
    train_ds = SSLAD2DDataset(
        img_dir=str(TRAIN_IMG_DIR),
        ann_file=str(TRAIN_ANN),
        img_size=img_size,
    )
    print(f"  train: {len(train_ds)} images")

    print("Loading labeled val dataset...")
    val_ds = SSLAD2DDataset(
        img_dir=str(VAL_IMG_DIR),
        ann_file=str(VAL_ANN),
        img_size=img_size,
    )
    print(f"  val: {len(val_ds)} images")

    print("Splitting unlabeled images into non-IID clients...")
    client_datasets = make_non_iid_clients_sslad(
        img_dir=str(UNLABELED_IMG_DIR),
        num_clients=args.num_clients,
        img_size=img_size,
        max_per_client=args.max_per_client,
    )

    # ---- DataLoaders ----
    server_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=labeled_yolo_collate, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Models ----
    use_pretrained = not args.no_pretrained
    server = Server(model=build_yolo(num_classes, pretrained=use_pretrained), loader=server_loader, cfg=cfg)
    global_model = build_yolo(num_classes, pretrained=use_pretrained)

    clients = []
    for cid, ds in enumerate(client_datasets):
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=unlabeled_yolo_collate, drop_last=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        clients.append(Client(
            client_id=cid,
            model=build_yolo(num_classes, pretrained=use_pretrained),
            loader=loader,
            cfg=cfg,
            num_samples=len(ds),
        ))

    def eval_hook(model, rnd, phase):
        return evaluate(model, val_ds, cfg.device, rnd, phase)

    trainer = FedSTO(
        global_model=global_model,
        server=server,
        clients=clients,
        cfg=cfg,
        eval_fn=eval_hook,
    )
    trainer.run()


if __name__ == "__main__":
    main()
