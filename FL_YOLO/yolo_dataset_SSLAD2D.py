"""SSLAD-2D 전용 데이터셋 로더
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence
import json

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


SSLAD_CLASSES = ("Pedestrian", "Cyclist", "Car", "Truck", "Tram", "Tricycle")
NUM_SSLAD_CLASSES = 6

CITY_GROUPS = {
    "SH": 0,
    "BJ": 1,
    "TY": 2,
    "GZ": 3,
    "SZ": 4,
}
DEFAULT_CITY_GROUP = 5  # 나머지 소도시 그룹


def _extract_city_code(filename: str) -> str:
    parts = filename.split("_")
    if filename.startswith("UNLABEL"):
        return parts[2] if len(parts) >= 3 else "UNK"
    elif filename.startswith("HT_"):
        return parts[3] if len(parts) >= 4 else "UNK"
    return "UNK"

class SSLAD2DDataset(Dataset):

    def __init__(self, img_dir: str, ann_file: str, img_size: int = 640):
        self.img_dir = Path(img_dir)
        self.img_size = img_size

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # image id와 메타데이터를 연결한다.
        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.id_to_idx = {img["id"]: i for i, img in enumerate(self.images)}

        # 주석 정보를 image_id별로 모은다.
        self.anns_by_img: Dict[int, list] = {img["id"]: [] for img in self.images}
        for ann in coco["annotations"]:
            if ann["image_id"] in self.anns_by_img:
                self.anns_by_img[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info["file_name"]

        # 이미지를 불러와 크기를 맞춘다.
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)

        # 정규화된 YOLO 형식 target을 만든다.
        anns = self.anns_by_img[img_info["id"]]
        if len(anns) == 0:
            targets = torch.zeros(0, 5)
        else:
            boxes = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                cls = ann["category_id"] - 1
                cx = (x + w / 2) / orig_w
                cy = (y + h / 2) / orig_h
                nw = w / orig_w
                nh = h / orig_h
                # 좌표 범위를 유효한 값으로 제한한다.
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                boxes.append([cls, cx, cy, nw, nh])
            targets = torch.tensor(boxes, dtype=torch.float32)

        return {"images": img_tensor, "targets": targets}


class SSLAD2DUnlabeled(Dataset):
    """SSLAD-2D unlabeled 이미지를 읽는 데이터셋."""

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
        return {"images": img_tensor}


def labeled_yolo_collate(batch):
    """이미지를 쌓고 (N, 6) 형식 target 텐서를 만든다."""
    imgs = torch.stack([b["images"] for b in batch]) # (N, 3, H, W)
    pieces = [] # 각 이미지의 타겟을 [img_idx, cls, cx, cy, w, h] 형식으로 변환하여 pieces 리스트에 추가
    for i, b in enumerate(batch): # batch의 각 요소에 대해 반복하면서, 타겟이 존재하는 경우 해당 타겟을 [img_idx, cls, cx, cy, w, h] 형식으로 변환하여 pieces 리스트에 추가
        t = b["targets"]
        if t.numel() == 0:
            continue
        idx = torch.full((t.shape[0], 1), float(i))
        pieces.append(torch.cat([idx, t], dim=1))
    targets = torch.cat(pieces, dim=0) if pieces else torch.zeros(0, 6)
    return {"images": imgs, "targets": targets}


def unlabeled_yolo_collate(batch):
    return {"images": torch.stack([b["images"] for b in batch])}



# 도시 코드 기반 non-IID 클라이언트 분할

def make_non_iid_clients_sslad(
    img_dir: str,
    num_clients: int = 5,
    img_size: int = 640,
    max_per_client: int | None = None,
) -> List[SSLAD2DUnlabeled]:
    """도시 코드 기준으로 unlabeled 이미지를 non-IID 클라이언트로 나눈다."""
    img_path = Path(img_dir)
    all_files = sorted(
        f.name for f in img_path.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    # 도시 그룹별로 파일을 모은다.
    buckets: Dict[int, List[str]] = {}
    for fname in all_files:
        code = _extract_city_code(fname)
        group = CITY_GROUPS.get(code, DEFAULT_CITY_GROUP)
        buckets.setdefault(group, []).append(fname) #buckets에 그룹별 파일 리스트 추가

    # 요청한 num_clients 수에 맞게 그룹을 병합하거나 분할한다.
    sorted_groups = sorted(buckets.keys()) # 그룹 키를 정렬하여 일관된 순서로 처리 (0, 1, 2, 3, 4, 5)
    client_file_lists: List[List[str]] = [] # 최종적으로 각 클라이언트에 할당할 파일 리스트를 담을 리스트 (각 요소는 한 클라이언트의 파일 리스트)

    if num_clients <= len(sorted_groups): # 요청된 클라이언트 수가 그룹 수보다 적으면, 작은 그룹들을 마지막 클라이언트로 병합
        # 남는 그룹은 마지막 클라이언트에 합친다.
        for i, gid in enumerate(sorted_groups): 
            if i < num_clients - 1:
                client_file_lists.append(buckets[gid])
            else:
                if i == num_clients - 1:
                    client_file_lists.append([])
                client_file_lists[-1].extend(buckets[gid])
    else: 
        # 클라이언트 수가 더 많으면 큰 그룹을 분할한다.
        for gid in sorted_groups:
            client_file_lists.append(buckets[gid])
        # 가장 큰 그룹을 나눠 남은 클라이언트를 채운다.
        while len(client_file_lists) < num_clients:
            biggest_idx = max(range(len(client_file_lists)),
                             key=lambda i: len(client_file_lists[i]))
            files = client_file_lists[biggest_idx]
            mid = len(files) // 2
            client_file_lists[biggest_idx] = files[:mid]
            client_file_lists.append(files[mid:])

    # 클라이언트별 최대 개수 제한을 적용한다.
    if max_per_client is not None:
        client_file_lists = [fl[:max_per_client] for fl in client_file_lists]

    datasets = []
    for fl in client_file_lists:
        datasets.append(SSLAD2DUnlabeled(img_dir, file_list=fl, img_size=img_size))

    print(f"Non-IID client split: {len(datasets)} clients, "
          f"images per client: {[len(d) for d in datasets]}")
    return datasets


# 합성 데이터 보조 함수
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
    """빠른 동작 확인용 합성 non-IID 클라이언트를 만든다."""
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
