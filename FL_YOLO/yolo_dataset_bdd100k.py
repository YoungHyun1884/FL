"""BDD100K 전용 데이터셋 로더
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


BDD100K_CLASSES = ("person", "car", "truck", "bus", "traffic sign")
NUM_BDD100K_CLASSES = 5

# 클라이언트 0: Overcast unlabeled
# 클라이언트 1: Rainy unlabeled
# 클라이언트 2: Snowy unlabeled
WEATHER_NAMES = {0: "Overcast", 1: "Rainy", 2: "Snowy"}

# Letterbox 채움 색상
_LETTERBOX_COLOR = (114, 114, 114)


def letterbox_image(
    img: Image.Image, target_size: int = 640,
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    
    orig_w, orig_h = img.size
    ratio = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    canvas = Image.new("RGB", (target_size, target_size), _LETTERBOX_COLOR)
    canvas.paste(img, (pad_w, pad_h))
    return canvas, ratio, (pad_w, pad_h)


def letterbox_targets(
    targets: torch.Tensor,
    orig_w: int, orig_h: int,
    ratio: float,
    pad_w: int, pad_h: int,
    target_size: int = 640,
) -> torch.Tensor:
    """YOLO 정규화 target을 letterbox 좌표계로 변환한다."""
    if targets.numel() == 0:
        return targets
    targets = targets.clone()
    # cx, cy는 원본 정규화 좌표를 픽셀 좌표로 바꾼 뒤 스케일과 패딩을 반영한다.
    targets[:, 1] = (targets[:, 1] * orig_w * ratio + pad_w) / target_size
    targets[:, 2] = (targets[:, 2] * orig_h * ratio + pad_h) / target_size
    # w, h는 스케일만 반영한다.
    targets[:, 3] = targets[:, 3] * orig_w * ratio / target_size
    targets[:, 4] = targets[:, 4] * orig_h * ratio / target_size
    targets[:, 1:].clamp_(0.0, 1.0)
    return targets


class YOLOAugment:

    def __init__(self, img_size: int = 640):
        self.img_size = img_size
        self.color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,
        )
        self.gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

    def __call__(
        self, img: Image.Image, targets: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor]:
        # 원본 target이 바뀌지 않도록 복사한다.
        if targets.numel() > 0:
            targets = targets.clone()

        # 1. 대규모 jittering
        if random.random() < 0.5:
            scale = random.uniform(0.5, 1.5)
            orig_w, orig_h = img.size
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img = TF.center_crop(img, (orig_h, orig_w))

            if targets.numel() > 0:
                # resize와 center-crop 이후 좌표 보정을 계산한다.
                off_x = (new_w - orig_w) / 2.0
                off_y = (new_h - orig_h) / 2.0

                # 정규화 좌표를 보정한 뒤 다시 원본 기준 정규화 좌표로 바꾼다.
                targets[:, 1] = (targets[:, 1] * new_w - off_x) / orig_w
                targets[:, 2] = (targets[:, 2] * new_h - off_y) / orig_h
                targets[:, 3] = targets[:, 3] * scale
                targets[:, 4] = targets[:, 4] * scale

                # 범위를 [0, 1]로 제한하고 화면 밖으로 많이 나간 박스를 제거한다.
                targets[:, 1:].clamp_(0.0, 1.0)
                # 중심점이 아직 화면 안에 있는 박스만 유지한다.
                valid = (targets[:, 1] > 0.01) & (targets[:, 1] < 0.99) & \
                        (targets[:, 2] > 0.01) & (targets[:, 2] < 0.99) & \
                        (targets[:, 3] > 0.005) & (targets[:, 4] > 0.005)
                targets = targets[valid]

        # 2. 좌우 반전
        if random.random() < 0.5:
            img = TF.hflip(img)
            if targets.numel() > 0:
                targets[:, 1] = 1.0 - targets[:, 1]

        # 3. 색상 변화
        if random.random() < 0.5:
            img = self.color_jitter(img)

        # 4. 그레이스케일
        if random.random() < 0.1:
            img = TF.to_grayscale(img, num_output_channels=3)

        # 5. 가우시안 블러
        if random.random() < 0.1:
            img = self.gaussian_blur(img)

        # 6. Cutout은 픽셀에만 적용하고 target은 그대로 둔다.
        if random.random() < 0.3:
            img_tensor = TF.to_tensor(img)
            eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))
            img_tensor = eraser(img_tensor)
            img = TF.to_pil_image(img_tensor)

        return img, targets


class BDD100KLabeled(Dataset):
    """YOLO txt annotation을 사용하는 BDD100K labeled 데이터셋."""

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 640,
        augment: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.aug_fn = YOLOAugment(img_size) if augment else None

        # 라벨 파일이 함께 있는 이미지 파일만 수집한다.
        self.files: List[str] = []
        for f in sorted(self.img_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                label_path = self.label_dir / (f.stem + ".txt")
                if label_path.exists():
                    self.files.append(f.stem)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        stem = self.files[idx]
        img_path = self.img_dir / (stem + ".jpg")
        label_path = self.label_dir / (stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

       
        targets = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    orig_cls = int(parts[0])
                    if orig_cls not in _TXT_CLASS_REMAP:
                        continue  
                    cls = _TXT_CLASS_REMAP[orig_cls]
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    targets.append([cls, cx, cy, w, h])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros(0, 5)

        if self.aug_fn is not None:
            img, targets = self.aug_fn(img, targets)
            orig_w, orig_h = img.size

        img, ratio, (pad_w, pad_h) = letterbox_image(img, self.img_size)
        targets = letterbox_targets(targets, orig_w, orig_h, ratio, pad_w, pad_h, self.img_size)
        img_tensor = TF.to_tensor(img)

        return {"images": img_tensor, "targets": targets}



# unlabeled 데이터셋

class BDD100KUnlabeled(Dataset):

    def __init__(
        self,
        img_dir: str,
        file_list: List[str] | None = None,
        img_size: int = 640,
        augment: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.augment = augment
        self.aug_fn = YOLOAugment(img_size) if augment else None

        if file_list is not None:
            self.files = sorted(file_list)
        else:
            self.files = sorted(
                f.name for f in self.img_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img_path = self.img_dir / self.files[idx]
        img = Image.open(img_path).convert("RGB")

        if self.aug_fn is not None:
            dummy_targets = torch.zeros(0, 5)
            img, _ = self.aug_fn(img, dummy_targets)

        img, _, _ = letterbox_image(img, self.img_size)
        img_tensor = TF.to_tensor(img)
        return {"images": img_tensor}



# collate 함수

def labeled_yolo_collate(batch):

    imgs = torch.stack([b["images"] for b in batch])
    pieces = []
    for i, b in enumerate(batch):
        t = b["targets"]
        if t.numel() == 0:
            continue
        idx = torch.full((t.shape[0], 1), float(i))
        pieces.append(torch.cat([idx, t], dim=1))
    targets = torch.cat(pieces, dim=0) if pieces else torch.zeros(0, 6)
    return {"images": imgs, "targets": targets}


def unlabeled_yolo_collate(batch):
    return {"images": torch.stack([b["images"] for b in batch])}



_WEATHER_TO_CLIENT = {
    "overcast": 0,
    "rainy": 1,
    "snowy": 2,
}


def make_weather_clients_bdd100k(
    data_root: str,
    split: str = "train",
    num_clients: int = 3,
    img_size: int = 640,
    max_per_client: int | None = None,
    augment: bool = True,
) -> List[BDD100KUnlabeled]:
    
    root = Path(data_root)
    img_dir = str(root / "images" / "100k" / split)
    ann_file = root / "labels" / f"bdd100k_labels_images_{split}.json"

    with open(ann_file, "r") as f:
        all_entries = json.load(f)

   
    client_files: Dict[int, List[str]] = {i: [] for i in range(num_clients)}
    for entry in all_entries:
        weather = entry.get("attributes", {}).get("weather", "unknown")
        cid = _WEATHER_TO_CLIENT.get(weather)
        if cid is None:
            continue  
        cid = min(cid, num_clients - 1)
        client_files[cid].append(entry["name"])

    datasets = []
    for cid in range(num_clients):
        files = sorted(client_files[cid])
        if max_per_client is not None and len(files) > max_per_client:
            random.shuffle(files)
            files = files[:max_per_client]
        ds = BDD100KUnlabeled(img_dir, file_list=files, img_size=img_size, augment=augment)
        datasets.append(ds)

    print(f"Weather-based client split: {len(datasets)} clients")
    for i, ds in enumerate(datasets):
        name = WEATHER_NAMES.get(i, f"client_{i}")
        print(f"  Client {i} ({name}): {len(ds)} images")

    return datasets


_BDD_CAT_TO_IDX = {
    "person": 0, "car": 1, "truck": 2, "bus": 3, "traffic sign": 4,
}

_TXT_CLASS_REMAP = {0: 0, 2: 1, 3: 2, 4: 3, 7: 4}




_SERVER_WEATHER = {"clear", "partly cloudy"}


def make_server_labeled_bdd100k(
    data_root: str,
    split: str = "train",
    img_size: int = 640,
    max_images: int | None = None,
    augment: bool = True,
    weather_filter: set | None = None,
):
    
    root = Path(data_root)
    img_dir = root / "images" / "100k" / split
    ann_file = root / "labels" / f"bdd100k_labels_images_{split}.json"

    if split == "train":
        wf = weather_filter if weather_filter is not None else _SERVER_WEATHER
    else:
        wf = None  # val에서는 날씨 필터를 적용하지 않는다.

    ds = BDD100KJsonLabeled(
        img_dir=str(img_dir),
        ann_file=str(ann_file),
        img_size=img_size,
        augment=augment,
        weather_filter=wf,
    )
    weather_str = f"weather={wf}" if wf else "all weather"
    print(f"  Server labeled ({split}): {len(ds)} images ({weather_str})")

    if max_images is not None and len(ds) > max_images:
        indices = list(range(len(ds)))
        random.shuffle(indices)
        ds = Subset(ds, indices[:max_images])
    return ds


class BDD100KJsonLabeled(Dataset):
    
    def __init__(
        self,
        img_dir: str,
        ann_file: str,
        img_size: int = 640,
        augment: bool = False,
        weather_filter: set | None = None,
    ):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.augment = augment
        self.aug_fn = YOLOAugment(img_size) if augment else None

        with open(ann_file, "r") as f:
            all_entries = json.load(f)

        # 필요하면 날씨 기준으로 먼저 걸러낸다.
        self.entries: List[dict] = []
        for entry in all_entries:
            if weather_filter is not None:
                w = entry.get("attributes", {}).get("weather", "unknown")
                if w not in weather_filter:
                    continue
            # detection label이 있는 항목만 남긴다.
            labels = entry.get("labels")
            if labels is None:
                continue
            # 논문에서 쓰는 5개 클래스의 box2d label만 사용한다.
            det_labels = [l for l in labels if "box2d" in l and l["category"] in _BDD_CAT_TO_IDX]
            if det_labels:
                self.entries.append({
                    "name": entry["name"],
                    "labels": det_labels,
                })

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        img_path = self.img_dir / entry["name"]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # BDD100K box2d 절대좌표를 YOLO 정규화 좌표로 변환한다.
        targets = []
        for lbl in entry["labels"]:
            box = lbl["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cls_idx = _BDD_CAT_TO_IDX[lbl["category"]]
            cx = ((x1 + x2) / 2.0) / orig_w
            cy = ((y1 + y2) / 2.0) / orig_h
            bw = (x2 - x1) / orig_w
            bh = (y2 - y1) / orig_h
            # 좌표 범위를 유효한 값으로 제한한다.
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.001, min(1.0, bw))
            bh = max(0.001, min(1.0, bh))
            targets.append([cls_idx, cx, cy, bw, bh])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros(0, 5)

        if self.aug_fn is not None:
            img, targets = self.aug_fn(img, targets)
            orig_w, orig_h = img.size

        img, ratio, (pad_w, pad_h) = letterbox_image(img, self.img_size)
        targets = letterbox_targets(targets, orig_w, orig_h, ratio, pad_w, pad_h, self.img_size)
        img_tensor = TF.to_tensor(img)

        return {"images": img_tensor, "targets": targets}
