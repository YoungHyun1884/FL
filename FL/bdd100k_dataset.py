"""BDD100K dataset loaders for FedSTO + YOLOv5.

Provides:
  - BDD100KLabeled:  labeled dataset (YOLO txt + JPEG) for server training/eval
  - BDD100KUnlabeled: unlabeled image-only dataset for client training
  - labeled_yolo_collate / unlabeled_yolo_collate: batch collate functions
  - make_weather_clients_bdd100k: split by weather condition for non-IID FL

BDD100K weather-split directory layout (pre-organized):
    train/{day,night,rain_day,rain_night,snowy_day,snowy_night}/
        images/*.jpg
        labels/*.txt   (YOLO format: class cx cy w h, normalized)

Following the FedSTO paper, clients are split by weather condition:
  - Client 0: Clear  (day)
  - Client 1: Overcast (night)
  - Client 2: Rainy  (rain_day + rain_night)

The server uses a separate labeled subset for supervised training.

BDD100K detection classes (10):
  0: pedestrian, 1: rider, 2: car, 3: truck, 4: bus,
  5: train, 6: motorcycle, 7: bicycle, 8: traffic light, 9: traffic sign
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
# Constants
# =====================================================================
BDD100K_CLASSES = (
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
)
NUM_BDD100K_CLASSES = 10

# Paper: 3 clients by weather condition
# Map weather-split folder names to client IDs
WEATHER_CLIENT_MAP = {
    "day": 0,           # Clear
    "night": 1,         # Overcast / Night
    "rain_day": 2,      # Rainy
    "rain_night": 2,    # Rainy (merged)
    "snowy_day": 2,     # grouped with Rainy as adverse weather
    "snowy_night": 2,   # grouped with Rainy as adverse weather
}

WEATHER_NAMES = {0: "Clear", 1: "Night", 2: "Adverse"}


# =====================================================================
# Data augmentation (paper: Mosaic, flip, jittering, graying,
#   gaussian blur, cutout, color space conversion)
# =====================================================================
class YOLOAugment:
    """Training-time augmentations following the FedSTO paper.

    Mosaic is omitted here (requires 4-image stitching inside the dataloader
    which conflicts with the standard Dataset/DataLoader pattern).
    The remaining 6 augmentations are applied stochastically.
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
        # Clone targets to avoid modifying the original
        if targets.numel() > 0:
            targets = targets.clone()

        # 1. Large-scale jittering (random resize + center-crop)
        if random.random() < 0.5:
            scale = random.uniform(0.5, 1.5)
            orig_w, orig_h = img.size
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img = TF.center_crop(img, (orig_h, orig_w))

            if targets.numel() > 0:
                # Targets are [cls, cx, cy, w, h] normalized to original image.
                # After resize by `scale` and center-crop back to original size:
                #   pixel_coord_in_crop = pixel_coord_in_resized - crop_offset
                # crop_offset = (new_dim - orig_dim) / 2
                off_x = (new_w - orig_w) / 2.0
                off_y = (new_h - orig_h) / 2.0

                # Convert normalized coords to pixel coords in resized image,
                # shift by crop offset, then normalize back to original size.
                targets[:, 1] = (targets[:, 1] * new_w - off_x) / orig_w
                targets[:, 2] = (targets[:, 2] * new_h - off_y) / orig_h
                targets[:, 3] = targets[:, 3] * scale  # width scales
                targets[:, 4] = targets[:, 4] * scale  # height scales

                # Clamp to [0, 1] and filter out boxes that went mostly off-screen
                targets[:, 1:].clamp_(0.0, 1.0)
                # Keep boxes whose center is still inside the image
                valid = (targets[:, 1] > 0.01) & (targets[:, 1] < 0.99) & \
                        (targets[:, 2] > 0.01) & (targets[:, 2] < 0.99) & \
                        (targets[:, 3] > 0.005) & (targets[:, 4] > 0.005)
                targets = targets[valid]

        # 2. Left-right flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            if targets.numel() > 0:
                targets[:, 1] = 1.0 - targets[:, 1]  # flip cx

        # 3. Color space conversion (ColorJitter)
        if random.random() < 0.5:
            img = self.color_jitter(img)

        # 4. Graying
        if random.random() < 0.1:
            img = TF.to_grayscale(img, num_output_channels=3)

        # 5. Gaussian blur
        if random.random() < 0.1:
            img = self.gaussian_blur(img)

        # 6. Cutout (random erase) — only applied to pixels, targets unchanged.
        #    Standard practice: cutout is an image-level regularization that
        #    does NOT remove ground-truth labels. The detector learns to be
        #    robust to partial occlusion (same as YOLOv5 default).
        if random.random() < 0.3:
            img_tensor = TF.to_tensor(img)
            eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))
            img_tensor = eraser(img_tensor)
            img = TF.to_pil_image(img_tensor)

        return img, targets


# =====================================================================
# Labeled dataset (server training / validation)
# =====================================================================
class BDD100KLabeled(Dataset):
    """BDD100K labeled dataset with YOLO-format txt annotations.

    Reads images and corresponding .txt label files.
    Returns YOLO-format targets: [cls, cx, cy, w, h] normalized.

    Args:
        img_dir:  path to image folder
        label_dir: path to label folder (.txt files, same name as images)
        img_size: resize images to (img_size, img_size)
        augment: whether to apply training augmentations
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

        # Collect image files that have matching label files
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

        # Parse YOLO labels: class cx cy w h (all normalized)
        targets = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    targets.append([cls, cx, cy, w, h])

        targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros(0, 5)

        # Apply augmentation before resize
        if self.aug_fn is not None:
            img, targets = self.aug_fn(img, targets)

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)

        return {"images": img_tensor, "targets": targets}


# =====================================================================
# Unlabeled dataset (client training - no labels used)
# =====================================================================
class BDD100KUnlabeled(Dataset):
    """BDD100K unlabeled image dataset for client semi-supervised training.

    Args:
        img_dir: path to image folder
        file_list: optional explicit list of image filenames
        img_size: resize images to (img_size, img_size)
        augment: whether to apply training augmentations
    """

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

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)
        return {"images": img_tensor}


# =====================================================================
# Collate functions (same as SSLAD, reusable)
# =====================================================================
def labeled_yolo_collate(batch):
    """Stack images and build (N, 6) target tensor [img_idx, cls, cx, cy, w, h]."""
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


# =====================================================================
# Weather-based client splitting (paper: Clear / Overcast / Rainy)
# =====================================================================
def make_weather_clients_bdd100k(
    data_root: str,
    split: str = "train",
    num_clients: int = 3,
    img_size: int = 640,
    max_per_client: int | None = None,
    augment: bool = True,
) -> List[BDD100KUnlabeled]:
    """Split BDD100K images by weather condition into non-IID clients.

    Following the FedSTO paper:
      Client 0: Clear (day)
      Client 1: Night/Overcast (night)
      Client 2: Adverse weather (rain_day + rain_night + snowy_day + snowy_night)

    Args:
        data_root: path to bdd100k root (contains train/, val/ folders)
        split: "train" or "val"
        num_clients: number of FL clients (default 3 per paper)
        img_size: resize target
        max_per_client: cap images per client (for debugging)
        augment: apply training augmentations

    Returns:
        List of BDD100KUnlabeled datasets, one per client.
    """
    root = Path(data_root) / split
    weather_dirs = ["day", "night", "rain_day", "rain_night", "snowy_day", "snowy_night"]

    # Bucket image paths by client ID
    client_buckets: Dict[int, List[Tuple[str, str]]] = {i: [] for i in range(num_clients)}

    for wdir in weather_dirs:
        img_dir = root / wdir / "images"
        if not img_dir.exists():
            continue
        cid = WEATHER_CLIENT_MAP.get(wdir, num_clients - 1)
        cid = min(cid, num_clients - 1)  # clamp if fewer clients requested
        files = sorted(f.name for f in img_dir.iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for fname in files:
            client_buckets[cid].append((str(img_dir), fname))

    datasets = []
    for cid in range(num_clients):
        entries = client_buckets[cid]
        if max_per_client is not None:
            entries = entries[:max_per_client]

        # Group by img_dir to create per-directory datasets, then combine
        dir_groups: Dict[str, List[str]] = {}
        for img_dir_str, fname in entries:
            dir_groups.setdefault(img_dir_str, []).append(fname)

        # Create a multi-directory unlabeled dataset
        ds = BDD100KUnlabeledMultiDir(dir_groups, img_size=img_size, augment=augment)
        datasets.append(ds)

    weather_info = {v: k for k, v in WEATHER_NAMES.items()}
    print(f"Weather-based client split: {len(datasets)} clients")
    for i, ds in enumerate(datasets):
        name = WEATHER_NAMES.get(i, f"client_{i}")
        print(f"  Client {i} ({name}): {len(ds)} images")

    return datasets


class BDD100KUnlabeledMultiDir(Dataset):
    """Unlabeled dataset spanning multiple directories (for weather-merged clients)."""

    def __init__(
        self,
        dir_files: Dict[str, List[str]],
        img_size: int = 640,
        augment: bool = False,
    ):
        self.img_size = img_size
        self.augment = augment
        self.aug_fn = YOLOAugment(img_size) if augment else None

        # Flatten to list of (dir_path, filename)
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
        return {"images": img_tensor}


# =====================================================================
# Server labeled dataset from weather splits (combined)
# =====================================================================
def make_server_labeled_bdd100k(
    data_root: str,
    split: str = "train",
    img_size: int = 640,
    max_images: int | None = None,
    augment: bool = True,
) -> BDD100KLabeledMultiDir:
    """Create a labeled dataset combining all weather splits for server training.

    Args:
        data_root: path to bdd100k root
        split: "train" or "val"
        img_size: resize target
        max_images: cap total images (for debugging)
        augment: apply training augmentations
    """
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
        indices = list(range(len(ds)))
        random.shuffle(indices)
        ds = Subset(ds, indices[:max_images])
    return ds


class BDD100KLabeledMultiDir(Dataset):
    """Labeled dataset spanning multiple weather-split directories."""

    def __init__(
        self,
        dir_pairs: List[Tuple[str, str]],
        img_size: int = 640,
        augment: bool = False,
    ):
        self.img_size = img_size
        self.augment = augment
        self.aug_fn = YOLOAugment(img_size) if augment else None

        # Collect all (img_path, label_path) pairs
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
