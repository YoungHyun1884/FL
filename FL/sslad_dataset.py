"""SSLAD-2D dataset loaders for FedSTO + YOLOv5.

  - 서버 쪽은 COCO 라벨을 읽어 YOLO 학습 형식으로 변환한다.
  - 클라이언트 쪽은 이미지만 읽고, 라벨은 teacher가 나중에 만든다.
  - collate 단계에서 YOLO loss가 요구하는 [img_idx, cls, cx, cy, w, h]
    형태로 batch 타깃을 합친다.

SSLAD-2D annotation format (COCO-style):
  bbox = [x, y, w, h]  (absolute pixels, top-left origin)
  category_id = 1..6    (1-indexed)

YOLO target format expected by ComputeLoss:
  [img_idx, cls, cx, cy, w, h]  (all normalized 0-1, 0-indexed class)

"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# =====================================================================
# SSLAD-2D 상수
# =====================================================================
SSLAD_CLASSES = ("Pedestrian", "Cyclist", "Car", "Truck", "Tram", "Tricycle")
NUM_SSLAD_CLASSES = 6

# unlabeled 파일명에 들어 있는 도시 코드 -> 클라이언트 그룹 매핑
# 주요 도시는 개별 클라이언트로 두고, 소도시는 함께 묶는다.
CITY_GROUPS = {
    "SH": 0,   # 상하이
    "BJ": 1,   # 베이징
    "TY": 2,   # 타이위안
    "GZ": 3,   # 광저우
    "SZ": 4,   # 선전
}
DEFAULT_CITY_GROUP = 5  # 나머지 소도시 그룹 (LZ, CD, YZ, TS, HN, TX, HB)


def _extract_city_code(filename: str) -> str:
    """SSLAD-2D 파일명에서 2글자 도시 코드를 추출한다.

    Labeled:   HT_TRAIN_000001_SH_000.jpg  -> SH
    Unlabeled: UNLABEL_01_BJ_000_00000000072719.jpg -> BJ
    """
    parts = filename.split("_")
    if filename.startswith("UNLABEL"):
        return parts[2] if len(parts) >= 3 else "UNK"
    elif filename.startswith("HT_"):
        return parts[3] if len(parts) >= 4 else "UNK"
    return "UNK"


# =====================================================================
# 레이블 데이터셋 (서버 학습 / 검증 / 테스트)
# =====================================================================
class SSLAD2DDataset(Dataset):
    """SSLAD-2D 레이블 데이터셋이다.

    COCO 형식 JSON 어노테이션을 읽고 JPEG 이미지를 즉시 불러온다.
    반환 형식은 정규화된 YOLO 타깃 [cls, cx, cy, w, h]이다.

    Args:
        img_dir:  이미지 폴더 경로 (예: .../labeled/train/)
        ann_file: 어노테이션 JSON 경로 (예: .../annotations/instance_train.json)
        img_size: 이미지를 (img_size, img_size)로 리사이즈
    """

    def __init__(self, img_dir: str, ann_file: str, img_size: int = 640):
        self.img_dir = Path(img_dir)
        self.img_size = img_size

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # image id -> 메타데이터 매핑을 만든다.
        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.id_to_idx = {img["id"]: i for i, img in enumerate(self.images)}

        # annotation을 image_id 기준으로 묶는다.
        self.anns_by_img: Dict[int, list] = {img["id"]: [] for img in self.images}
        for ann in coco["annotations"]:
            if ann["image_id"] in self.anns_by_img:
                self.anns_by_img[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info["file_name"]

        # 1) 원본 이미지를 읽고 YOLO 입력 크기로 맞춘다.
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)  # (3, H, W), float32 [0, 1]

        # 2) COCO bbox를 YOLO target으로 바꾼다.
        #    COCO는 좌상단 기준 [x, y, w, h]이고,
        #    YOLO loss는 중심좌표 기준 [cls, cx, cy, w, h]를 사용한다.
        anns = self.anns_by_img[img_info["id"]]
        if len(anns) == 0:
            targets = torch.zeros(0, 5)
        else:
            boxes = []
            for ann in anns:
                x, y, w, h = ann["bbox"]  # COCO: 절대좌표 [x, y, w, h]
                cls = ann["category_id"] - 1  # 1부터 시작 -> 0부터 시작
                cx = (x + w / 2) / orig_w
                cy = (y + h / 2) / orig_h
                nw = w / orig_w
                nh = h / orig_h
                # 학습 안정성을 위해 좌표를 0~1 범위로 제한한다.
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                boxes.append([cls, cx, cy, nw, nh])
            targets = torch.tensor(boxes, dtype=torch.float32)

        return {"images": img_tensor, "targets": targets}


# =====================================================================
# 비라벨 데이터셋 (클라이언트 학습)
# =====================================================================
class SSLAD2DUnlabeled(Dataset):
    """SSLAD-2D 비라벨 이미지 데이터셋이다.

    디렉토리 또는 지정된 경로 목록에서 JPEG 이미지를 불러온다.

    Args:
        img_dir:   unlabeled 이미지 폴더 경로
        file_list: 파일명 목록을 직접 지정할 때 사용하는 선택 인자 (non-IID 분할용)
        img_size:  이미지를 (img_size, img_size)로 리사이즈
    """

    def __init__(
        self,
        img_dir: str,
        file_list: List[str] | None = None,
        img_size: int = 640,
    ):
        self.img_dir = Path(img_dir)
        self.img_size = img_size

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
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)
        # 비라벨 데이터는 정답 박스 없이 이미지만 반환한다.
        # 이후 클라이언트의 EMA teacher가 이 이미지를 보고 가짜 타깃을 만든다.
        return {"images": img_tensor}


# =====================================================================
# Collate 함수 (YOLO 형식)
# =====================================================================
def labeled_yolo_collate(batch):
    """이미지를 쌓고 [img_idx, cls, cx, cy, w, h] 형태의 (N, 6) 타깃 텐서를 만든다."""
    imgs = torch.stack([b["images"] for b in batch])  # (B, 3, H, W)
    pieces = []
    for i, b in enumerate(batch):
        t = b["targets"]
        if t.numel() == 0:
            continue
        # 한 batch 안에서 각 박스가 몇 번째 이미지 소속인지 기록해야
        # YOLO ComputeLoss가 target을 올바른 이미지 예측과 매칭할 수 있다.
        idx = torch.full((t.shape[0], 1), float(i))
        pieces.append(torch.cat([idx, t], dim=1))
    targets = torch.cat(pieces, dim=0) if pieces else torch.zeros(0, 6)
    return {"images": imgs, "targets": targets}


def unlabeled_yolo_collate(batch):
    # 비라벨 배치는 이미지 텐서만 묶으면 된다.
    return {"images": torch.stack([b["images"] for b in batch])}


# =====================================================================
# 도시 코드 기준 non-IID 클라이언트 분할
# =====================================================================
def make_non_iid_clients_sslad(
    img_dir: str,
    num_clients: int = 5,
    img_size: int = 640,
    max_per_client: int | None = None,
) -> List[SSLAD2DUnlabeled]:
    """도시 코드를 기준으로 unlabeled 이미지를 non-IID 클라이언트로 나눈다.

    주요 도시(SH, BJ, TY, GZ, SZ)는 각각 하나의 클라이언트가 된다.
    num_clients > 5이면 소도시는 추가 클라이언트 하나로 묶는다.
    num_clients < 도시 그룹 수이면 여러 그룹을 합친다.

    Args:
        img_dir: unlabeled 이미지 폴더 경로 (예: .../unlabel/image_0/)
        num_clients: 원하는 FL 클라이언트 수
        img_size: 이미지 리사이즈 목표 크기
        max_per_client: 클라이언트당 이미지 수 제한값 (개발/디버깅용)

    Returns:
        클라이언트별 SSLAD2DUnlabeled 데이터셋 리스트
    """
    img_path = Path(img_dir)
    all_files = sorted(
        f.name for f in img_path.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    # 1) 파일명 속 도시 코드를 읽어 client bucket을 만든다.
    #    이렇게 하면 client마다 도시별 편향이 생겨 non-IID 환경이 된다.
    buckets: Dict[int, List[str]] = {}
    for fname in all_files:
        code = _extract_city_code(fname)
        group = CITY_GROUPS.get(code, DEFAULT_CITY_GROUP)
        buckets.setdefault(group, []).append(fname)

    # 2) 요청한 client 수와 bucket 수가 다를 수 있으므로 합치거나 나눈다.
    sorted_groups = sorted(buckets.keys())
    client_file_lists: List[List[str]] = []

    if num_clients <= len(sorted_groups):
        # 클라이언트 수가 부족하면 뒤쪽 그룹들을 마지막 클라이언트에 합친다.
        for i, gid in enumerate(sorted_groups):
            if i < num_clients - 1:
                client_file_lists.append(buckets[gid])
            else:
                if i == num_clients - 1:
                    client_file_lists.append([])
                client_file_lists[-1].extend(buckets[gid])
    else:
        # 클라이언트 수가 더 많으면 가장 큰 bucket을 계속 반으로 나눠 채운다.
        for gid in sorted_groups:
            client_file_lists.append(buckets[gid])
        while len(client_file_lists) < num_clients:
            biggest_idx = max(range(len(client_file_lists)),
                             key=lambda i: len(client_file_lists[i]))
            files = client_file_lists[biggest_idx]
            mid = len(files) // 2
            client_file_lists[biggest_idx] = files[:mid]
            client_file_lists.append(files[mid:])

    # 3) 디버깅할 때는 client당 이미지 수를 제한할 수 있다.
    if max_per_client is not None:
        client_file_lists = [fl[:max_per_client] for fl in client_file_lists]

    datasets = []
    for fl in client_file_lists:
        # 최종 결과는 "각 클라이언트가 자기 이미지 목록만 보도록 만든 비라벨 dataset"이다.
        datasets.append(SSLAD2DUnlabeled(img_dir, file_list=fl, img_size=img_size))

    print(f"Non-IID 클라이언트 분할: {len(datasets)}개 클라이언트, "
          f"클라이언트별 이미지 수: {[len(d) for d in datasets]}")
    return datasets


# =====================================================================
# 합성 데이터 보조 함수 (실데이터 없이 빠른 스모크 테스트용)
# =====================================================================
class SyntheticYOLODataset(Dataset):
    def __init__(
        self,
        n: int,
        img_size: int = 320,
        num_classes: int = 6,
        max_boxes: int = 4,
        min_boxes: int = 1,
        brightness: float = 0.0,
        class_weights: Sequence[float] | None = None,
        labeled: bool = True,
        seed: int = 0,
    ):
        self.n = n
        self.img_size = img_size
        self.num_classes = num_classes
        self.labeled = labeled

        g = torch.Generator().manual_seed(seed)
        base = torch.rand(n, 3, img_size, img_size, generator=g)
        self.imgs = (base + brightness).clamp(0, 1)

        if class_weights is None:
            class_weights = [1.0] * num_classes
        cw = torch.tensor(class_weights, dtype=torch.float)
        cw = cw / cw.sum()

        self.targets: List[torch.Tensor] = []
        for i in range(n):
            k = int(torch.randint(min_boxes, max_boxes + 1, (1,), generator=g).item())
            cls = torch.multinomial(cw, k, replacement=True, generator=g).float()
            cx = torch.rand(k, generator=g) * 0.6 + 0.2
            cy = torch.rand(k, generator=g) * 0.6 + 0.2
            bw = torch.rand(k, generator=g) * 0.2 + 0.1
            bh = torch.rand(k, generator=g) * 0.2 + 0.1
            self.targets.append(torch.stack([cls, cx, cy, bw, bh], dim=1))

            for j in range(k):
                c = int(cls[j].item())
                x1 = int((cx[j] - bw[j] / 2).clamp(0, 1) * img_size)
                x2 = int((cx[j] + bw[j] / 2).clamp(0, 1) * img_size)
                y1 = int((cy[j] - bh[j] / 2).clamp(0, 1) * img_size)
                y2 = int((cy[j] + bh[j] / 2).clamp(0, 1) * img_size)
                channel = c % 3
                self.imgs[i, channel, y1:y2, x1:x2] = (
                    self.imgs[i, channel, y1:y2, x1:x2] * 0.3 + 0.7
                )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        item = {"images": self.imgs[i]}
        if self.labeled:
            item["targets"] = self.targets[i]
        return item


def make_non_iid_clients(
    num_clients: int,
    samples_per_client: int,
    num_classes: int = 6,
    img_size: int = 320,
    base_seed: int = 100,
) -> List[SyntheticYOLODataset]:
    """스모크 테스트용 합성 non-IID 클라이언트를 만든다."""
    clients = []
    shifts = [-0.15, +0.1, +0.25, -0.3, +0.2]
    for k in range(num_clients):
        cw = [1.0] * num_classes
        dominant = k % num_classes
        cw[dominant] = 4.0
        cw[(dominant + 1) % num_classes] = 2.0
        ds = SyntheticYOLODataset(
            n=samples_per_client,
            img_size=img_size,
            num_classes=num_classes,
            brightness=shifts[k % len(shifts)],
            class_weights=cw,
            labeled=False,
            seed=base_seed + k,
        )
        clients.append(ds)
    return clients
