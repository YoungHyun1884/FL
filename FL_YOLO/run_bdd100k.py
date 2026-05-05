"""BDD100K 데이터셋과 YOLOv5L 모델을 사용하여 FedSTO 알고리즘을 실행하는 메인 파일.
- BDD100K 데이터셋을 날씨별로 나누어 3개의 non-IID 클라이언트로 사용
- 서버는 모든 날씨에서 labeled 데이터를 가지고 있으며, 클라이언트는 unlabeled 데이터만 가지고 있음
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
from .yolo_dataset_bdd100k import (
    NUM_BDD100K_CLASSES,
    labeled_yolo_collate,
    unlabeled_yolo_collate,
    make_weather_clients_bdd100k,
    make_server_labeled_bdd100k,
)


DATA_ROOT = Path("/home/pyh/바탕화면/FL/dataset/BDD100k")


@torch.no_grad()
def evaluate(model: YOLOv5Detector, val_ds, device: str, rnd: int, phase: str,
             num_workers: int = 0):
    """분리된 labeled validation set에서 평균 supervised loss를 계산한다.

    평가가 학습 상태를 오염시키지 않도록 BN running stat을 저장했다가 복원한다.
    """
    import copy
    loader = DataLoader(val_ds, batch_size=16, collate_fn=labeled_yolo_collate,
                        num_workers=num_workers, pin_memory=(num_workers > 0))

    # 평가 전에 BN running stat을 저장한다.
    bn_state = {}
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            bn_state[name] = (m.running_mean.clone(), m.running_var.clone(),
                              m.num_batches_tracked.clone())

    model.eval()
    model.yolo.train()  # ComputeLoss는 train 모드의 raw output을 필요로 한다.
    total, n = 0.0, 0
    for batch in loader:
        imgs = batch["images"].to(device)
        tgts = batch["targets"].to(device)
        loss_dict = model.supervised_loss(imgs, tgts)
        total += float(sum(loss_dict.values()).detach())
        n += 1
        if n >= 50:  # 속도를 위해 상한을 둔다.
            break

    # BN running stat을 복원한다.
    for name, m in model.named_modules():
        if name in bn_state:
            m.running_mean.copy_(bn_state[name][0])
            m.running_var.copy_(bn_state[name][1])
            m.num_batches_tracked.copy_(bn_state[name][2])

    mean_loss = total / max(n, 1)
    print(f"  [eval@{phase}:r{rnd}] val_det_loss={mean_loss:.4f}")
    return {"val_loss": mean_loss}


def build_yolo(num_classes: int, pretrained: bool = True):
    """YOLOv5L 기반 탐지기를 생성한다."""
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
    parser.add_argument("--warmup-rounds", type=int, default=30)
    parser.add_argument("--t1", type=int, default=70,
                        help="Phase 1 rounds (paper: 100)")
    parser.add_argument("--t2", type=int, default=120,
                        help="Phase 2 rounds (paper: 150)")
    parser.add_argument("--local-epochs", type=int, default=1,
                        help="Local epochs per round (paper: 1)")
    parser.add_argument("--server-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip COCO pretrained weights (random init)")
    parser.add_argument("--max-per-client", type=int, default=5000,
                        help="Cap images per client (paper: ~5K per client)")
    parser.add_argument("--max-server-images", type=int, default=5000,
                        help="Cap server labeled images (paper: ~5K)")
    parser.add_argument("--fedavg-alpha", type=float, default=0.0,
                        help="Blending: global = alpha*old + (1-alpha)*FedAvg (0.0 = pure FedAvg, paper default)")
    parser.add_argument("--fedavg-alpha-nonbackbone", type=float, default=0.95,
                        help="Phase 2 neck/head blending alpha (default: 0.95)")
    parser.add_argument("--exclude-bn", action="store_true",
                        help="Exclude BN running stats from FedAvg (paper includes by default)")
    parser.add_argument("--phase1-hard-only", action="store_true",
                        help="Use only hard pseudo labels during Phase 1")
    parser.add_argument("--phase1-soft-weight", type=float, default=1.0,
                        help="Additional multiplier for Phase 1 soft pseudo loss")
    parser.add_argument("--phase1-lr-scale", type=float, default=1.0,
                        help="Scale client LR during Phase 1 only")
    parser.add_argument("--phase1-max-batches", type=int, default=None,
                        help="Optional max client batches per epoch during Phase 1")
    parser.add_argument("--resume-warmup", type=str, default=None,
                        help="Path to warmup checkpoint to skip warmup")
    parser.add_argument("--resume-phase1", type=str, default=None,
                        help="Path to phase1 checkpoint to skip warmup+phase1, run phase2 only")
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
        tau_low_start=0.1, tau_high_start=0.15,
        use_epoch_adaptor=True,
        unsup_loss_weight=4.0,              # 논문 기본값 복원
        server_epoch=True,                  # 논문: 서버도 1 epoch per round
        fedavg_alpha=args.fedavg_alpha,
        fedavg_alpha_nonbackbone=args.fedavg_alpha_nonbackbone,
        fedavg_exclude_bn=args.exclude_bn,
        phase1_hard_only=args.phase1_hard_only,
        phase1_soft_weight=args.phase1_soft_weight,
        phase1_client_lr_scale=args.phase1_lr_scale,
        phase1_max_batches_per_epoch=args.phase1_max_batches,
        ortho_lambda=1e-4,
        ortho_power_iters=1,
        ema_decay=0.999,
        server_opt=OptimConfig(lr=1e-3, momentum=0.937, weight_decay=5e-4),
        client_opt=OptimConfig(lr=1e-3, momentum=0.937, weight_decay=5e-4),
        device="cuda" if torch.cuda.is_available() else "cpu",
        ckpt_dir="./checkpoints_bdd100k",
        log_every=1,
    )
    print(f"device = {cfg.device}")
    n_gpus = torch.cuda.device_count() if str(cfg.device).startswith("cuda") else 0
    if n_gpus > 1:
        print(f"client devices = {[f'cuda:{i}' for i in range(n_gpus)]}")
    print(
        "Phase1 ablation:",
        f"hard_only={cfg.phase1_hard_only}",
        f"soft_weight={cfg.phase1_soft_weight}",
        f"lr_scale={cfg.phase1_client_lr_scale}",
        f"max_batches={cfg.phase1_max_batches_per_epoch}",
    )

    num_classes = NUM_BDD100K_CLASSES  # 논문에서 사용하는 5개 클래스
    img_size = args.img_size
    data_root = args.data_root

    # ---- 서버 labeled 데이터셋 ----
    print("Loading server labeled dataset (train, Cloudy weather)...")
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

    # ---- 클라이언트 unlabeled 데이터셋 ----
    print("Splitting unlabeled images by weather into non-IID clients...")
    client_datasets = make_weather_clients_bdd100k(
        data_root=data_root, split="train",
        num_clients=args.num_clients, img_size=img_size,
        max_per_client=args.max_per_client, augment=True,
    )

    # ---- 데이터 로더 ----
    server_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=labeled_yolo_collate, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- 모델 ----
    use_pretrained = not args.no_pretrained
    print(f"Building YOLOv5L (pretrained={use_pretrained}, nc={num_classes})...")
    server = Server(model=build_yolo(num_classes, pretrained=use_pretrained),
                    loader=server_loader, cfg=cfg)
    global_model = build_yolo(num_classes, pretrained=use_pretrained)

    clients = []
    for cid, ds in enumerate(client_datasets):
        client_device = f"cuda:{(cid + 1) % n_gpus}" if n_gpus > 1 else cfg.device
        generator = None
        if n_gpus > 1:
            generator = torch.Generator()
            generator.manual_seed(cfg.seed + 1000 + cid)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=unlabeled_yolo_collate, drop_last=True,
            num_workers=args.num_workers, pin_memory=True,
            generator=generator,
        )
        clients.append(Client(
            client_id=cid,
            model=build_yolo(num_classes, pretrained=use_pretrained),
            loader=loader,
            cfg=cfg,
            num_samples=len(ds),
            device=client_device,
        ))

    nw = args.num_workers
    def eval_hook(model, rnd, phase):
        return evaluate(model, val_ds, cfg.device, rnd, phase, num_workers=nw)

    trainer = FedSTO(
        global_model=global_model,
        server=server,
        clients=clients,
        cfg=cfg,
        eval_fn=eval_hook,
    )
    trainer.run(
        skip_warmup=args.resume_warmup is not None or args.resume_phase1 is not None,
        warmup_ckpt=args.resume_warmup,
        skip_phase1=args.resume_phase1 is not None,
        phase1_ckpt=args.resume_phase1,
    )


if __name__ == "__main__":
    main()
