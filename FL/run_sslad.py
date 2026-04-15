"""
    FedSTO + YOLOv5 on SSLAD-2D
    1) 서버는 labeled train을 사용해 먼저 warmup을 수행한다.
    2) 각 client는 자기 unlabeled 이미지에 대해 local EMA teacher로
       pseudo label을 만든 뒤 student를 업데이트한다.
    3) client 업데이트를 서버가 FedAvg로 합친다.
    4) 서버는 labeled 데이터로 한 번 더 supervised refinement를 수행한다.
    5) 위 과정을 Phase 1/2 규칙에 맞게 반복한다.

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
from .sslad_dataset import (
    SSLAD2DDataset,
    NUM_SSLAD_CLASSES,
    labeled_yolo_collate,
    unlabeled_yolo_collate,
    make_non_iid_clients_sslad,
)


# =====================================================================
# 기본 경로 (필요하면 CLI로 덮어쓸 수 있음)
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
    """분리된 레이블 검증셋에서 평균 지도 손실을 계산한다."""
    # 검증은 항상 레이블 validation set으로만 수행한다.
    # 즉, 가짜 라벨 품질을 직접 보는 대신 "현재 전역 모델이
    # 실제 정답 라벨에 대해 얼마나 잘 맞는지"를 loss로 추적한다.
    loader = DataLoader(val_ds, batch_size=8, collate_fn=labeled_yolo_collate,
                        num_workers=4, pin_memory=True)
    model.eval()
    model.yolo.train()  # ComputeLoss는 train 모드의 raw output을 사용한다.
    total, n = 0.0, 0
    for batch in loader:
        imgs = batch["images"].to(device)
        tgts = batch["targets"].to(device)
        loss_dict = model.supervised_loss(imgs, tgts)
        total += float(sum(loss_dict.values()).detach())
        n += 1
        if n >= 50:  # 속도를 위해 상한을 둔다.
            break
    mean_loss = total / max(n, 1)
    print(f"  [eval@{phase}:r{rnd}] val_det_loss={mean_loss:.4f}")
    return {"val_loss": mean_loss}


def build_yolo(num_classes: int, pretrained: bool = True):
    """YOLOv5Detector 인스턴스를 만드는 함수다. 기본값은 COCO 사전학습이다."""
    # 모든 서버 / 클라이언트 / 전역 모델은 같은 YOLO 래퍼를 사용한다.
    # 차이는 "어느 데이터로 어떤 손실을 계산하느냐"에 있다.
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

    # ---- 데이터셋 ----
    # 레이블 데이터는 서버 전용이다.
    # train_ds: 사전학습 + 각 라운드 이후 서버 지도 보정용
    # val_ds: 전역 모델 성능 확인용
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
    # 비라벨 데이터는 파일명 속 city code 기준으로 클라이언트별로 나눈다.
    # 따라서 각 클라이언트는 서로 다른 분포를 가진 non-IID 환경을 흉내 내게 된다.
    client_datasets = make_non_iid_clients_sslad(
        img_dir=str(UNLABELED_IMG_DIR),
        num_clients=args.num_clients,
        img_size=img_size,
        max_per_client=args.max_per_client,
    )

    # ---- 데이터로더 ----
    # 서버는 레이블 loader를, 각 클라이언트는 비라벨 loader를 가진다.
    server_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=labeled_yolo_collate, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- 모델 ----
    use_pretrained = not args.no_pretrained
    # server.model:
    #   서버가 레이블 데이터로 직접 업데이트하는 모델
    # global_model:
    #   클라이언트들에게 전파되는 전역 기준 모델
    # client.model:
    #   각 클라이언트에서 local EMA teacher와 함께 학습되는 student 모델
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

    # FedSTO.run() 안에서 실제 데이터 흐름은 다음 순서로 진행된다.
    # 사전학습 -> 클라이언트 로컬 가짜 라벨 학습 -> 집계
    # -> 서버 레이블 보정 -> 평가
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
