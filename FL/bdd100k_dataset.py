"""BDD100K 전용 데이터셋 로더
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import random

import torch
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# =====================================================================
# 상수
# =====================================================================
BDD100K_CLASSES = (
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
)
NUM_BDD100K_CLASSES = 10

# 논문 기준: 날씨 조건별 3개 클라이언트
# 날씨 분할 폴더 이름을 클라이언트 ID에 매핑한다.
WEATHER_CLIENT_MAP = {
    "day": 0,           
    "night": 1,       
    "rain_day": 2,   
    "rain_night": 2,    
    "snowy_day": 2,     
    "snowy_night": 2,   
}

WEATHER_NAMES = {0: "Clear", 1: "Night", 2: "Adverse"}


# =====================================================================
# 데이터 증강 (논문: Mosaic, flip, jittering, graying,
#   gaussian blur, cutout, color space conversion)
# =====================================================================
class YOLOAugment:
    """FedSTO 논문을 따르는 학습 시점 증강이다.

    Mosaic은 여기서 제외했다. dataloader 안에서 4장 이미지를 붙여야 하는데
    일반적인 Dataset/DataLoader 패턴과 충돌하기 때문이다.
    나머지 6개 증강은 확률적으로 적용한다.
    """

    def __init__(self, img_size: int = 640):
        self.img_size = img_size
        self.color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,
        )
        self.gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

    def __call__(
        self, img: Image.Image, targets: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor]:
        # 증강 과정에서 좌표를 바꾸므로 원본 target은 건드리지 않는다.
        if targets.numel() > 0:
            targets = targets.clone()

        # 1) 큰 스케일 흔들기:
        #    이미지를 크게/작게 바꾼 뒤 다시 원래 시야로 crop해서
        #    스케일 변화에 강하게 만든다.
        if random.random() < 0.5:
            scale = random.uniform(0.5, 1.5)
            orig_w, orig_h = img.size
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img = TF.center_crop(img, (orig_h, orig_w))

            if targets.numel() > 0:
                # target은 원본 기준 정규화 좌표이므로,
                # resize + center crop 이후에는 좌표도 같은 방식으로 다시 계산해야 한다.
                off_x = (new_w - orig_w) / 2.0
                off_y = (new_h - orig_h) / 2.0

                targets[:, 1] = (targets[:, 1] * new_w - off_x) / orig_w
                targets[:, 2] = (targets[:, 2] * new_h - off_y) / orig_h
                targets[:, 3] = targets[:, 3] * scale
                targets[:, 4] = targets[:, 4] * scale

                # 화면 밖으로 거의 밀려난 박스는 제거해준다.
                targets[:, 1:].clamp_(0.0, 1.0)
                valid = (targets[:, 1] > 0.01) & (targets[:, 1] < 0.99) & \
                        (targets[:, 2] > 0.01) & (targets[:, 2] < 0.99) & \
                        (targets[:, 3] > 0.005) & (targets[:, 4] > 0.005)
                targets = targets[valid]

        # 2) 좌우 반전
        if random.random() < 0.5:
            img = TF.hflip(img)
            if targets.numel() > 0:
                targets[:, 1] = 1.0 - targets[:, 1]

        # 3) 색감 변화
        if random.random() < 0.5:
            img = self.color_jitter(img)

        # 4) 흑백 변환
        if random.random() < 0.1:
            img = TF.to_grayscale(img, num_output_channels=3)

        # 5) 블러
        if random.random() < 0.1:
            img = self.gaussian_blur(img)

        # 6) cutout은 픽셀만 가리고 라벨은 유지한다.
        #    일부가 가려져도 물체를 찾도록 만드는 정규화 역할이다.
        if random.random() < 0.3:
            img_tensor = TF.to_tensor(img)
            eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))
            img_tensor = eraser(img_tensor)
            img = TF.to_pil_image(img_tensor)

        return img, targets


# =====================================================================
# 레이블 데이터셋 (서버 학습 / 검증)
# =====================================================================
class BDD100KLabeled(Dataset):
    """YOLO 형식의 txt 어노테이션을 사용하는 BDD100K 레이블 데이터셋이다.

    이미지와 대응되는 .txt 라벨 파일을 읽는다.
    반환 형식은 정규화된 YOLO 타깃 [cls, cx, cy, w, h]이다.

    Args:
        img_dir:  이미지 폴더 경로
        label_dir: 라벨 폴더 경로 (.txt 파일, 이미지와 같은 이름)
        img_size: 이미지를 (img_size, img_size)로 리사이즈
        augment: 학습용 증강 적용 여부
    """

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

        # 대응되는 라벨 파일이 있는 이미지 파일만 모은다.
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

        # BDD100K는 이미 YOLO txt 포맷이라서
        # SSLAD처럼 COCO bbox를 다시 바꿀 필요 없이 바로 읽으면 된다.
        targets = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    targets.append([cls, cx, cy, w, h])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros(0, 5)

        # augmentation은 resize 전에 적용해서 좌표도 함께 바꿔준다.
        if self.aug_fn is not None:
            img, targets = self.aug_fn(img, targets)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)

        return {"images": img_tensor, "targets": targets}


# =====================================================================
# 비라벨 데이터셋 (클라이언트 학습 - 라벨 미사용)
# =====================================================================
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
            # 비라벨 경로는 정답이 없으므로 dummy target만 넣고
            # 증강에서는 이미지 쪽 변화만 적용한다.
            dummy_targets = torch.zeros(0, 5)
            img, _ = self.aug_fn(img, dummy_targets)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)
        # 클라이언트에서는 이 이미지만 teacher에게 들어가 가짜 라벨이 만들어진다.
        return {"images": img_tensor}


# =====================================================================
# Collate 함수 (SSLAD와 동일하게 재사용 가능)
# =====================================================================
def labeled_yolo_collate(batch): #개별 img_tensor를 모아 하나의 배치로 묶는다.
    imgs = torch.stack([b["images"] for b in batch])
    pieces = []
    for i, b in enumerate(batch):
        t = b["targets"]
        if t.numel() == 0:
            continue
        # YOLO loss가 batch 안에서 각 박스의 소속 이미지를 알아야 하므로 img_idx를 앞에 붙여 하나의 target 텐서로 합친다.
        idx = torch.full((t.shape[0], 1), float(i))
        pieces.append(torch.cat([idx, t], dim=1))
    targets = torch.cat(pieces, dim=0) if pieces else torch.zeros(0, 6)
    return {"images": imgs, "targets": targets}


def unlabeled_yolo_collate(batch):
    # unlabeled 쪽은 이미지 텐서만 묶으면 충분하다.
    return {"images": torch.stack([b["images"] for b in batch])}


# =====================================================================
# 날씨 기반 클라이언트 분할 (논문: 맑음 / 흐림 / 비)
# =====================================================================
def make_weather_clients_bdd100k(
    data_root: str,
    split: str = "train",
    num_clients: int = 3,
    img_size: int = 640,
    max_per_client: int | None = None,
    augment: bool = True,
) -> List[BDD100KUnlabeled]:
    
    root = Path(data_root) / split
    weather_dirs = ["day", "night", "rain_day", "rain_night", "snowy_day", "snowy_night"]

    # 1) 날씨 폴더를 클라이언트 id에 매핑해 non-IID 버킷을 만든다.
    client_buckets: Dict[int, List[Tuple[str, str]]] = {i: [] for i in range(num_clients)}

    for wdir in weather_dirs:
        img_dir = root / wdir / "images"
        if not img_dir.exists():
            continue
        cid = WEATHER_CLIENT_MAP.get(wdir, num_clients - 1)
        cid = min(cid, num_clients - 1)  # 요청된 클라이언트 수가 적으면 범위를 맞춘다.
        files = sorted(f.name for f in img_dir.iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for fname in files:
            client_buckets[cid].append((str(img_dir), fname)) #client_buckets에 (이미지 디렉토리, 파일명) 쌍을 저장한다.

    datasets = []
    for cid in range(num_clients): # num_clients에 저장된 이미지 목록을 client dataset으로 만든다.
        entries = client_buckets[cid] 
        if max_per_client is not None:
            entries = entries[:max_per_client]

        # 같은 클라이언트 안에서도 rain_day / rain_night처럼 폴더가 여러 개일 수 있어서
        # 디렉토리별로 묶은 뒤 하나의 multi-dir dataset으로 합친다.
        dir_groups: Dict[str, List[str]] = {}
        for img_dir_str, fname in entries:
            dir_groups.setdefault(img_dir_str, []).append(fname) # dir_groups는 같은 client에 속한 이미지 디렉토리별로 파일명을 그룹화한다.

        ds = BDD100KUnlabeledMultiDir(dir_groups, img_size=img_size, augment=augment) 
        datasets.append(ds)

    weather_info = {v: k for k, v in WEATHER_NAMES.items()}
    print(f"Weather-based client split: {len(datasets)} clients")
    for i, ds in enumerate(datasets):
        name = WEATHER_NAMES.get(i, f"client_{i}")
        print(f"  Client {i} ({name}): {len(ds)} images")

    return datasets


class BDD100KUnlabeledMultiDir(Dataset): # 클라이언트용 비라벨 dataset으로, 같은 클라이언트의 여러 날씨 폴더 이미지를 하나로 묶어 반환한다.

    def __init__(
        self,
        dir_files: Dict[str, List[str]],
        img_size: int = 640,
        augment: bool = False,
    ):
        self.img_size = img_size
        self.augment = augment
        self.aug_fn = YOLOAugment(img_size) if augment else None

        # 여러 날씨 폴더의 파일 목록을 하나의 순회 가능한 리스트로 펼친다.
        self.entries: List[Tuple[str, str]] = []
        for dir_path, files in dir_files.items():
            for f in files:
                self.entries.append((dir_path, f))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        dir_path, fname = self.entries[idx]
        img_path = Path(dir_path) / fname
        img = Image.open(img_path).convert("RGB")

        if self.aug_fn is not None:
            dummy = torch.zeros(0, 5)
            img, _ = self.aug_fn(img, dummy)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)
        # 결국 클라이언트 입장에서는 "이미지 한 장"만 꺼내 쓰면 되므로
        # 디렉토리가 여러 개여도 반환 형식은 동일하다.
        return {"images": img_tensor}


# =====================================================================
# 날씨 분할을 합친 서버 레이블 데이터셋
# =====================================================================
def make_server_labeled_bdd100k(
    data_root: str,
    split: str = "train",
    img_size: int = 640,
    max_images: int | None = None,
    augment: bool = True,
) -> BDD100KLabeledMultiDir:
    """서버 학습용으로 모든 날씨 분할을 합친 레이블 데이터셋을 만든다.

    Args:
        data_root: bdd100k 루트 경로
        split: "train" 또는 "val"
        img_size: 리사이즈 목표 크기
        max_images: 전체 이미지 수 제한값 (디버깅용)
        augment: 학습용 증강 적용 여부
    """
    # 서버는 전체 날씨 분포를 보도록 모든 weather 폴더를 합쳐 레이블 dataset을 만든다.
    root = Path(data_root) / split
    weather_dirs = ["day", "night", "rain_day", "rain_night", "snowy_day", "snowy_night"]

    dir_pairs: List[Tuple[str, str]] = []  # (img_dir, label_dir)
    for wdir in weather_dirs:
        img_dir = root / wdir / "images"
        lbl_dir = root / wdir / "labels"
        if img_dir.exists() and lbl_dir.exists():
            dir_pairs.append((str(img_dir), str(lbl_dir)))

    ds = BDD100KLabeledMultiDir(dir_pairs, img_size=img_size, augment=augment)
    if max_images is not None and len(ds) > max_images:
        # 디버깅용으로 전체 레이블 pool에서 일부만 무작위 샘플링할 수 있다.
        indices = list(range(len(ds)))
        random.shuffle(indices)
        ds = Subset(ds, indices[:max_images])
    return ds


class BDD100KLabeledMultiDir(Dataset):
    """여러 날씨 분할 디렉토리를 아우르는 레이블 데이터셋이다."""

    def __init__(
        self,
        dir_pairs: List[Tuple[str, str]],
        img_size: int = 640,
        augment: bool = False,
    ):
        self.img_size = img_size
        self.augment = augment
        self.aug_fn = YOLOAugment(img_size) if augment else None

        # 여러 날씨 폴더의 (이미지, 라벨) 쌍을 하나의 리스트로 모은다.
        self.entries: List[Tuple[Path, Path]] = []
        for img_dir_str, lbl_dir_str in dir_pairs:
            img_dir = Path(img_dir_str)
            lbl_dir = Path(lbl_dir_str)
            for f in sorted(img_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    lbl_path = lbl_dir / (f.stem + ".txt")
                    if lbl_path.exists():
                        self.entries.append((f, lbl_path))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        img_path, label_path = self.entries[idx]
        img = Image.open(img_path).convert("RGB")

        # label txt는 이미 YOLO 형식이므로 그대로 읽어 tensor로 만든다.
        targets = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    targets.append([cls, cx, cy, w, h])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros(0, 5)

        if self.aug_fn is not None:
            img, targets = self.aug_fn(img, targets)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)

        return {"images": img_tensor, "targets": targets}
