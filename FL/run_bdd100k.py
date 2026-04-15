"""End-to-end FedSTO run with YOLOv5L on BDD100K.

Pipeline matches the FedSTO paper (NeurIPS 2023):
    Warmup (50 rounds, server supervised on labeled data)
      -> Phase 1: Selective Training (100 rounds, backbone only)
      -> Phase 2: FPT + Orthogonal Enhancement (150 rounds)

Clients are split by weather condition (non-IID):
    Client 0: Clear (day)
    Client 1: Night
    Client 2: Adverse (rain + snow)

Dataset layout expected under DATA_ROOT:
    train/{day,night,rain_day,rain_night,snowy_day,snowy_night}/
        images/*.jpg
        labels/*.txt   (YOLO format: class cx cy w h)
    val/{day,night,...}/
        images/*.jpg
        labels/*.txt
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
from .bdd100k_dataset import (
    NUM_BDD100K_CLASSES,
    labeled_yolo_collate,
    unlabeled_yolo_collate,
    make_weather_clients_bdd100k,
    make_server_labeled_bdd100k,
    BDD100KLabeledMultiDir,
)


DATA_ROOT = Path("/home/pyh/바탕화면/FL/dataset/BDD100k")


@torch.no_grad()
def evaluate(model: YOLOv5Detector, val_ds, device: str, rnd: int, phase: str):
    """Evaluate: mean supervised loss on the held-out labeled val set."""
    loader = DataLoader(val_ds, batch_size=16, collate_fn=labeled_yolo_collate,
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
    """Factory for YOLOv5L Detector (COCO-pretrained by default)."""
    return YOLOv5Detector(
        num_classes=num_classes,
        tau_high=0.6,
        tau_low=0.1,
        iou_thres=0.65,
        pretrained=pretrained,
    )


def main():
    parser = argparse.ArgumentParser(description="FedSTO + YOLOv5L on BDD100K")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT),
                        help="Path to BDD100K root directory")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for server/client loaders")
    parser.add_argument("--num-clients", type=int, default=3,
                        help="Number of FL clients (3 = weather split per paper)")
    parser.add_argument("--warmup-rounds", type=int, default=50)
    parser.add_argument("--t1", type=int, default=100,
                        help="Phase 1 rounds (paper: 100)")
    parser.add_argument("--t2", type=int, default=150,
                        help="Phase 2 rounds (paper: 150)")
    parser.add_argument("--local-epochs", type=int, default=1,
                        help="Local epochs per round (paper: 1)")
    parser.add_argument("--server-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip COCO pretrained weights (random init)")
    parser.add_argument("--max-per-client", type=int, default=None,
                        help="Cap images per client (for debugging)")
    parser.add_argument("--max-server-images", type=int, default=None,
                        help="Cap server labeled images (for debugging)")
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
        ckpt_dir="./checkpoints_bdd100k",
        log_every=1,
    )
    print(f"device = {cfg.device}")

    num_classes = NUM_BDD100K_CLASSES  # 10
    img_size = args.img_size
    data_root = args.data_root

    # ---- Server labeled dataset (all weather splits combined) ----
    print("Loading server labeled dataset (train, all weather splits)...")
    train_ds = make_server_labeled_bdd100k(
        data_root=data_root, split="train", img_size=img_size,
        max_images=args.max_server_images, augment=True,
    )
    print(f"  server train: {len(train_ds)} images")

    print("Loading validation dataset...")
    val_ds = make_server_labeled_bdd100k(
        data_root=data_root, split="val", img_size=img_size, augment=False,
    )
    print(f"  val: {len(val_ds)} images")

    # ---- Client unlabeled datasets (split by weather) ----
    print("Splitting unlabeled images by weather into non-IID clients...")
    client_datasets = make_weather_clients_bdd100k(
        data_root=data_root, split="train",
        num_clients=args.num_clients, img_size=img_size,
        max_per_client=args.max_per_client, augment=True,
    )

    # ---- DataLoaders ----
    server_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=labeled_yolo_collate, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Models ----
    use_pretrained = not args.no_pretrained
    print(f"Building YOLOv5L (pretrained={use_pretrained}, nc={num_classes})...")
    server = Server(model=build_yolo(num_classes, pretrained=use_pretrained),
                    loader=server_loader, cfg=cfg)
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
