"""SSLAD-2D dataset loaders for FedSTO + YOLOv5.

Provides:
  - SSLAD2DDataset: labeled dataset (COCO JSON + JPEG) for server training/eval
  - SSLAD2DUnlabeled: unlabeled image-only dataset for client training
  - labeled_yolo_collate / unlabeled_yolo_collate: batch collate functions
  - make_non_iid_clients_sslad: split unlabeled images by city code for non-IID FL

SSLAD-2D annotation format (COCO-style):
  bbox = [x, y, w, h]  (absolute pixels, top-left origin)
  category_id = 1..6    (1-indexed)

YOLO target format expected by ComputeLoss:
  [img_idx, cls, cx, cy, w, h]  (all normalized 0-1, 0-indexed class)

Also retains the original synthetic helpers for quick smoke-testing.
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
# SSLAD-2D constants
# =====================================================================
SSLAD_CLASSES = ("Pedestrian", "Cyclist", "Car", "Truck", "Tram", "Tricycle")
NUM_SSLAD_CLASSES = 6

# City codes found in unlabeled filenames -> client group mapping
# Major cities get their own client; minor cities are grouped together.
CITY_GROUPS = {
    "SH": 0,   # Shanghai
    "BJ": 1,   # Beijing
    "TY": 2,   # Taiyuan
    "GZ": 3,   # Guangzhou
    "SZ": 4,   # Shenzhen
}
DEFAULT_CITY_GROUP = 5  # all other minor cities (LZ, CD, YZ, TS, HN, TX, HB)


def _extract_city_code(filename: str) -> str:
    """Extract 2-letter city code from SSLAD-2D filename.

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
# Labeled dataset (server training / validation / test)
# =====================================================================
class SSLAD2DDataset(Dataset):
    """SSLAD-2D labeled dataset.

    Reads COCO-format JSON annotation and loads JPEG images on-the-fly.
    Returns YOLO-format targets: [cls, cx, cy, w, h] normalized.

    Args:
        img_dir:  path to image folder (e.g. .../labeled/train/)
        ann_file: path to annotation JSON (e.g. .../annotations/instance_train.json)
        img_size: resize images to (img_size, img_size)
    """

    def __init__(self, img_dir: str, ann_file: str, img_size: int = 640):
        self.img_dir = Path(img_dir)
        self.img_size = img_size

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # Build image id -> metadata mapping
        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.id_to_idx = {img["id"]: i for i, img in enumerate(self.images)}

        # Group annotations by image_id
        self.anns_by_img: Dict[int, list] = {img["id"]: [] for img in self.images}
        for ann in coco["annotations"]:
            if ann["image_id"] in self.anns_by_img:
                self.anns_by_img[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info["file_name"]

        # Load and resize image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)  # (3, H, W), float32 [0, 1]

        # Build YOLO targets: [cls, cx, cy, w, h] normalized
        anns = self.anns_by_img[img_info["id"]]
        if len(anns) == 0:
            targets = torch.zeros(0, 5)
        else:
            boxes = []
            for ann in anns:
                x, y, w, h = ann["bbox"]  # COCO: absolute [x, y, w, h]
                cls = ann["category_id"] - 1  # 1-indexed -> 0-indexed
                cx = (x + w / 2) / orig_w
                cy = (y + h / 2) / orig_h
                nw = w / orig_w
                nh = h / orig_h
                # Clamp to valid range
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                boxes.append([cls, cx, cy, nw, nh])
            targets = torch.tensor(boxes, dtype=torch.float32)

        return {"images": img_tensor, "targets": targets}


# =====================================================================
# Unlabeled dataset (client training)
# =====================================================================
class SSLAD2DUnlabeled(Dataset):
    """SSLAD-2D unlabeled image dataset.

    Loads JPEG images from a directory (or a list of specific paths).

    Args:
        img_dir:   path to unlabeled image folder
        file_list: optional explicit list of filenames (for non-IID splitting)
        img_size:  resize images to (img_size, img_size)
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
        return {"images": img_tensor}


# =====================================================================
# Collate functions (YOLO format)
# =====================================================================
def labeled_yolo_collate(batch):
    """Stack images and build (N, 6) target tensor [img_idx, cls, cx, cy, w, h]."""
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


# =====================================================================
# Non-IID client splitting by city code
# =====================================================================
def make_non_iid_clients_sslad(
    img_dir: str,
    num_clients: int = 5,
    img_size: int = 640,
    max_per_client: int | None = None,
) -> List[SSLAD2DUnlabeled]:
    """Split unlabeled images into non-IID clients by city code.

    Each major city (SH, BJ, TY, GZ, SZ) becomes one client.
    Minor cities are grouped into an extra client if num_clients > 5.
    If num_clients < number of city groups, groups are merged.

    Args:
        img_dir: path to unlabeled image folder (e.g. .../unlabel/image_0/)
        num_clients: desired number of FL clients
        img_size: resize target for images
        max_per_client: cap images per client (useful for dev/debug)

    Returns:
        List of SSLAD2DUnlabeled datasets, one per client.
    """
    img_path = Path(img_dir)
    all_files = sorted(
        f.name for f in img_path.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    # Bucket files by city group
    buckets: Dict[int, List[str]] = {}
    for fname in all_files:
        code = _extract_city_code(fname)
        group = CITY_GROUPS.get(code, DEFAULT_CITY_GROUP)
        buckets.setdefault(group, []).append(fname) #buckets에 그룹별 파일 리스트 추가

    # Sort bucket keys and merge/split to match num_clients
    sorted_groups = sorted(buckets.keys()) # 그룹 키를 정렬하여 일관된 순서로 처리 (0, 1, 2, 3, 4, 5)
    client_file_lists: List[List[str]] = [] # 최종적으로 각 클라이언트에 할당할 파일 리스트를 담을 리스트 (각 요소는 한 클라이언트의 파일 리스트)

    if num_clients <= len(sorted_groups): # 요청된 클라이언트 수가 그룹 수보다 적으면, 작은 그룹들을 마지막 클라이언트로 병합
        # Merge smaller groups into the last client
        for i, gid in enumerate(sorted_groups): 
            if i < num_clients - 1:
                client_file_lists.append(buckets[gid])
            else:
                if i == num_clients - 1:
                    client_file_lists.append([])
                client_file_lists[-1].extend(buckets[gid])
    else: 
        # More clients requested than groups: split largest groups
        for gid in sorted_groups:
            client_file_lists.append(buckets[gid])
        # Fill remaining clients by splitting the largest bucket
        while len(client_file_lists) < num_clients:
            biggest_idx = max(range(len(client_file_lists)),
                             key=lambda i: len(client_file_lists[i]))
            files = client_file_lists[biggest_idx]
            mid = len(files) // 2
            client_file_lists[biggest_idx] = files[:mid]
            client_file_lists.append(files[mid:])

    # Apply per-client cap
    if max_per_client is not None:
        client_file_lists = [fl[:max_per_client] for fl in client_file_lists]

    datasets = []
    for fl in client_file_lists:
        datasets.append(SSLAD2DUnlabeled(img_dir, file_list=fl, img_size=img_size))

    print(f"Non-IID client split: {len(datasets)} clients, "
          f"images per client: {[len(d) for d in datasets]}")
    return datasets


# =====================================================================
# Synthetic helpers (kept for quick smoke-testing without real data)
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
    """Synthetic non-IID clients for smoke-testing."""
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
