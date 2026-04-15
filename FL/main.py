"""테스트 파일 : 랜덤으로 생성한 데이터셋과 더미 모델로 FedSTO 알고리즘이 돌아가는지 확인하는 파일.
실제 YOLO 모델과 SSLAD-2D 데이터셋이 없을 때, FedSTO의 전체적인 흐름이 돌아가는지 확인하기 위해 만든 테스트 파일.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, Dataset

from . import (
    FedSTOConfig, OptimConfig, DummyDetector,
    Client, Server, FedSTO,
)


# ---------- 합성 데이터셋 ----------
class SyntheticLabeled(Dataset):
    
    def __init__(self, n=512, num_classes=5, img=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.imgs = torch.randn(n, 3, img, img, generator=g)
        self.cls = torch.randint(0, num_classes, (n,), generator=g)
        self.xy = torch.rand(n, 2, generator=g)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        return {
            "images": self.imgs[i],
            "targets": {"cls": self.cls[i], "xy": self.xy[i]},
        }


class SyntheticUnlabeled(Dataset):
    """클라이언트마다 분포 편향이 다른 unlabeled 데이터를 만든다."""
    def __init__(self, n=512, img=16, shift=0.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.imgs = torch.randn(n, 3, img, img, generator=g) + shift

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        return {"images": self.imgs[i]}


def labeled_collate(batch):
    imgs = torch.stack([b["images"] for b in batch])
    cls = torch.stack([b["targets"]["cls"] for b in batch])
    xy = torch.stack([b["targets"]["xy"] for b in batch])
    return {"images": imgs, "targets": {"cls": cls, "xy": xy}}


def unlabeled_collate(batch):
    return {"images": torch.stack([b["images"] for b in batch])}


# ---------- 평가 훅 ----------
@torch.no_grad()
def evaluate(model, rnd, phase, val_ds=None):
    if val_ds is None:
        return {}
    model.eval()
    loader = DataLoader(val_ds, batch_size=64, collate_fn=labeled_collate)
    device = next(model.parameters()).device
    correct = total = 0
    for batch in loader:
        imgs = batch["images"].to(device)
        tgt = batch["targets"]["cls"].to(device)
        out = model(imgs)
        pred = out[:, :model.num_classes].argmax(-1)
        correct += (pred == tgt).sum().item()
        total += tgt.numel()
    acc = correct / max(total, 1)
    print(f"  [eval@{phase}:r{rnd}] cls_acc={acc:.4f}")
    model.train()
    return {"acc": acc}


def main():
    cfg = FedSTOConfig(
        warmup_rounds=2, #서버 사전 학습 에폭 수
        T1=5, T2=5, # Phase 1과 Phase 2의 라운드 수
        local_epochs=1, server_steps=10, # 클라이언트: 1 에폭/라운드, 서버: 스텝 수
        num_clients=3, 
        client_sample_ratio=1.0,# 역할 : 각 라운드마다 클라이언트 샘플링 비율 (1.0 = 모든 클라이언트)
        # DummyDetector는 softmax confidence가 크지 않아서 임계값을 낮게 둔다.
        tau_low=0.1, tau_high=0.4, # 임계값 : 클라이언트의 효율적인 교사 PLA에서 소프트 라벨과 하드 가짜 라벨을 구분하는 임계값
        tau_low_start=0.05, tau_high_start=0.2, # 초기 임계값 : 라운드 0에서 tau_low와 tau_high의 초기값 (각각 0.05와 0.2로 설정)
        use_epoch_adaptor=True, 
        unsup_loss_weight=3.0,
        ortho_lambda=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ckpt_dir="./checkpoints_demo",
        log_every=1,
    )
    print(f"device = {cfg.device}")

    num_classes = 5

    # 데이터셋
    labeled_ds = SyntheticLabeled(n=512, num_classes=num_classes, seed=0) # 서버의 레이블이 있는 데이터셋
    val_ds = SyntheticLabeled(n=128, num_classes=num_classes, seed=999) # 검증용 데이터셋 (서버와 동일한 분포에서 생성)
    client_datasets = [
        SyntheticUnlabeled(n=400, shift=+0.5, seed=1),  # client 0: 비 오는 분포
        SyntheticUnlabeled(n=400, shift=-0.5, seed=2),  # client 1: 눈 오는 분포
        SyntheticUnlabeled(n=400, shift=+1.0, seed=3),  # client 2: 흐린 날 분포
    ]

    server_loader = DataLoader(labeled_ds, batch_size=32, shuffle=True, collate_fn=labeled_collate)

    # 모델: 클라이언트별 모델, 서버 모델, 전역 기준 모델을 각각 만든다.
    def make_model():
        return DummyDetector(num_classes=num_classes)

    server_model = make_model()
    global_model = make_model()

    server = Server(model=server_model, loader=server_loader, cfg=cfg)

    clients = []
    for cid, ds in enumerate(client_datasets):
        loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=unlabeled_collate)
        clients.append(Client(
            client_id=cid,
            model=make_model(),
            loader=loader,
            cfg=cfg,
            num_samples=len(ds),
        ))

    def eval_hook(model, rnd, phase):
        return evaluate(model, rnd, phase, val_ds=val_ds)

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
