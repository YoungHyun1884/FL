"""FedSTO + YOLOv5L on BDD100K
Clients are split by weather condition (non-IID):
    Client 0: Clear (day)
    Client 1: Night
    Client 2: Adverse (rain + snow)

    1) 서버는 모든 날씨 폴더의 labeled 데이터를 합쳐서 warmup을 한다.
    2) client는 날씨별로 나뉜 자기 이미지 묶음만 본다.
    3) 각 client는 unlabeled처럼 이미지만 사용하고, local EMA teacher가
       pseudo label을 만들어 student를 학습시킨다.
    4) client 결과를 FedAvg로 합친 뒤, 서버가 다시 labeled 데이터로 보정한다.

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
    """분리된 레이블 검증셋에서 평균 지도 손실을 계산한다."""
    # 검증은 validation split의 정답 라벨을 그대로 사용한다.
    # 즉, 클라이언트 가짜 라벨 품질이 아니라 현재 전역 detector의 실제 성능을 본다.
    loader = DataLoader(val_ds, batch_size=16, collate_fn=labeled_yolo_collate,
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
    """YOLOv5L 탐지기 인스턴스를 만드는 함수다. 기본값은 COCO 사전학습이다."""
    # 서버 / 전역 / 클라이언트가 모두 같은 YOLO 래퍼를 쓰고,
    # 학습 데이터와 손실 경로만 역할에 따라 달라진다.
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

    # ---- 서버 labeled 데이터셋 (모든 날씨 분할 통합) ----
    # 서버는 특정 날씨 하나만 보지 않고, train 아래의 모든 날씨 폴더를 합쳐서 사용한다.
    # 그래서 클라이언트가 날씨 편향된 업데이트를 보내더라도 서버가 레이블 데이터로 균형을 잡아준다.
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

    # ---- 클라이언트 unlabeled 데이터셋 (날씨 기준 분할) ----
    # 클라이언트는 날씨별 non-IID 분포를 가지도록 나뉜다.
    # day / night / adverse(rain,snow) 조합이 각각 하나의 클라이언트 그룹이 된다.
    print("Splitting unlabeled images by weather into non-IID clients...")
    client_datasets = make_weather_clients_bdd100k(
        data_root=data_root, split="train",
        num_clients=args.num_clients, img_size=img_size,
        max_per_client=args.max_per_client, augment=True,
    )

    # ---- 데이터로더 ----
    # 서버 loader는 레이블 타깃을 포함하고,
    # 클라이언트 loader는 이미지 텐서만 묶어 반지도 경로로 보낸다.
    server_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=labeled_yolo_collate, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- 모델 ----
    use_pretrained = not args.no_pretrained
    print(f"Building YOLOv5L (pretrained={use_pretrained}, nc={num_classes})...")
    # server.model:
    #   레이블 데이터로 지도 보정을 수행하는 모델
    # global_model:
    #   각 라운드 시작 때 클라이언트에 전파되는 전역 모델
    # client.model:
    #   local EMA teacher와 함께 가짜 라벨 학습을 수행하는 student 모델
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

    # 실제 학습 흐름:
    # 서버 사전학습 -> 날씨별 클라이언트 로컬 학습 -> 집계
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
