import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from tqdm import tqdm


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def read_image(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# Grid pipeline used in the paper: zero-margin pad crop -> 2x2 tiles -> SimCLR.
# The archived experiment disables the optional supervised penalty below.
DATASET_DIR = os.environ.get("UAV_DATASET_DIR", r"path\to\2026_data_UAP_UAI")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CKPT = os.path.join(OUTPUT_DIR, "ssl_encoder_grid_hybrid.pth")

PAD_CLASSES = (2, 3)
HUMAN_VEHICLE_CLASSES = ()

BACKBONE = "resnet101"
IMG_SIZE = 224
PROJ_DIM = 128

BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.2
PENALTY_WEIGHT = 0.0
UNSAFE_MARGIN = 0.0

# Zero-margin keeps the landing pad texture isolated. Change to 0.10 only if
# you intentionally want contextual border pixels in both training and testing.
ROI_PAD_RATIO = 0.0
MIN_PAD_SIZE = 32
MIN_OBJ_INTERSECTION_RATIO = 0.0

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
USE_AMP = True
USE_CHANNELS_LAST = True
USE_RANDOM_CROP = False
USE_GAUSSIAN_BLUR = True
SAVE_LATEST_EACH_EPOCH = True
LATEST_CKPT = os.path.join(OUTPUT_DIR, "ssl_encoder_grid_hybrid_latest.pth")
INDEX_CACHE_PATH = os.path.join(OUTPUT_DIR, "ssl_grid_index_cache.npz")
USE_INDEX_CACHE = True
REBUILD_INDEX = True
SEED = 42


@dataclass(frozen=True)
class TileRef:
    img_path: str
    roi_xyxy: Tuple[int, int, int, int]
    tile_id: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def coords_are_normalized(vals: np.ndarray) -> bool:
    return np.all(vals >= 0.0) and np.all(vals <= 1.0)


def read_label_file(path: str) -> List[Tuple[int, float, float, float, float]]:
    if not os.path.exists(path):
        return []

    rows: List[Tuple[int, float, float, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            rows.append((cls, cx, cy, w, h))
    return rows


def yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    normalized: bool,
) -> Tuple[int, int, int, int]:
    if normalized:
        cx *= img_w
        w *= img_w
        cy *= img_h
        h *= img_h

    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = int(round(cx + w / 2.0))
    y2 = int(round(cy + h / 2.0))
    return x1, y1, x2, y2


def clamp_xyxy(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x1c = max(0, min(img_w - 1, x1))
    y1c = max(0, min(img_h - 1, y1))
    x2c = max(0, min(img_w, x2))
    y2c = max(0, min(img_h, y2))
    if x2c < x1c:
        x1c, x2c = x2c, x1c
    if y2c < y1c:
        y1c, y2c = y2c, y1c
    return x1c, y1c, x2c, y2c


def is_fully_visible(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> bool:
    return x1 >= 0 and y1 >= 0 and x2 <= img_w and y2 <= img_h and x2 > x1 and y2 > y1


def area_xyxy(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def has_human_or_vehicle(
    pad_xyxy: Tuple[int, int, int, int],
    objects_xyxy: List[Tuple[int, int, int, int]],
    min_intersection_ratio: float,
) -> bool:
    for obj in objects_xyxy:
        # Count a pad as dirty only when the object box is actually on the pad.
        if intersection_area(pad_xyxy, obj) > 0:
            return True
    return False


def expand_box(
    box: Tuple[int, int, int, int],
    ratio: float,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * ratio))
    pad_y = int(round(bh * ratio))
    ex1, ey1 = x1 - pad_x, y1 - pad_y
    ex2, ey2 = x2 + pad_x, y2 + pad_y
    if not is_fully_visible(ex1, ey1, ex2, ey2, img_w, img_h):
        return ()
    return clamp_xyxy(ex1, ey1, ex2, ey2, img_w, img_h)


def split_grid_2x2(img: np.ndarray) -> List[np.ndarray]:
    h, w = img.shape[:2]
    h2 = h // 2
    w2 = w // 2
    tiles = [
        img[0:h2, 0:w2],
        img[0:h2, w2:w],
        img[h2:h, 0:w2],
        img[h2:h, w2:w],
    ]
    return [tile for tile in tiles if tile.shape[0] >= 16 and tile.shape[1] >= 16]


def tile_boxes_2x2(roi_xyxy: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi_xyxy
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    return [
        (x1, y1, xm, ym),
        (xm, y1, x2, ym),
        (x1, ym, xm, y2),
        (xm, ym, x2, y2),
    ]


def dirty_tile_ids(
    roi_xyxy: Tuple[int, int, int, int],
    objects_xyxy: List[Tuple[int, int, int, int]],
) -> List[int]:
    ids: List[int] = []
    for tile_id, tile_box in enumerate(tile_boxes_2x2(roi_xyxy)):
        if any(intersection_area(tile_box, obj) > 0 for obj in objects_xyxy):
            ids.append(tile_id)
    return ids


def load_tile(ref: TileRef, img_size: int = IMG_SIZE) -> np.ndarray:
    img = read_image(ref.img_path)
    if img is None:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    x1, y1, x2, y2 = ref.roi_xyxy
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    tiles = split_grid_2x2(roi)
    if not tiles:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    tile = tiles[min(ref.tile_id, len(tiles) - 1)]
    return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)


class HybridGridPadDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform,
        pad_classes: Tuple[int, ...] = PAD_CLASSES,
        human_vehicle_classes: Tuple[int, ...] = HUMAN_VEHICLE_CLASSES,
        roi_pad_ratio: float = ROI_PAD_RATIO,
        min_pad_size: int = MIN_PAD_SIZE,
        min_obj_intersection_ratio: float = MIN_OBJ_INTERSECTION_RATIO,
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.pad_classes = set(pad_classes)
        self.hv_classes = set(human_vehicle_classes)
        self.roi_pad_ratio = roi_pad_ratio
        self.min_pad_size = min_pad_size
        self.min_obj_intersection_ratio = min_obj_intersection_ratio
        self.clean_index: List[TileRef] = []
        self.dirty_index: List[TileRef] = []
        self._build_index()

    def _load_index_cache(self) -> bool:
        if not USE_INDEX_CACHE or REBUILD_INDEX or not os.path.exists(INDEX_CACHE_PATH):
            return False

        try:
            cache = np.load(INDEX_CACHE_PATH, allow_pickle=False)
            if str(cache["images_dir"][0]) != self.images_dir:
                return False
            if str(cache["labels_dir"][0]) != self.labels_dir:
                return False
            if "human_vehicle_classes" not in cache:
                return False
            cached_hv_classes = tuple(int(x) for x in cache["human_vehicle_classes"])
            if cached_hv_classes != tuple(sorted(self.hv_classes)):
                return False
            if not np.isclose(float(cache["roi_pad_ratio"][0]), float(self.roi_pad_ratio)):
                return False
            if int(cache["min_pad_size"][0]) != int(self.min_pad_size):
                return False
            if not np.isclose(
                float(cache["min_obj_intersection_ratio"][0]),
                float(self.min_obj_intersection_ratio),
            ):
                return False

            clean_paths = cache["clean_paths"]
            clean_rois = cache["clean_rois"]
            clean_tile_ids = cache["clean_tile_ids"]
            dirty_paths = cache["dirty_paths"]
            dirty_rois = cache["dirty_rois"]
            dirty_tile_ids = cache["dirty_tile_ids"]

            self.clean_index = [
                TileRef(str(path), tuple(map(int, roi)), int(tile_id))
                for path, roi, tile_id in zip(clean_paths, clean_rois, clean_tile_ids)
            ]
            self.dirty_index = [
                TileRef(str(path), tuple(map(int, roi)), int(tile_id))
                for path, roi, tile_id in zip(dirty_paths, dirty_rois, dirty_tile_ids)
            ]

            if len(self.clean_index) < 100 or (PENALTY_WEIGHT > 0.0 and len(self.dirty_index) < 10):
                self.clean_index = []
                self.dirty_index = []
                return False

            print(
                f"[INDEX] loaded cache: clean={len(self.clean_index)} dirty={len(self.dirty_index)}",
                flush=True,
            )
            return True
        except Exception as exc:
            print(f"[INDEX] cache ignored: {exc}", flush=True)
            self.clean_index = []
            self.dirty_index = []
            return False

    def _save_index_cache(self) -> None:
        if not USE_INDEX_CACHE:
            return

        def refs_to_arrays(refs: List[TileRef]):
            if not refs:
                return (
                    np.array([], dtype=str),
                    np.empty((0, 4), dtype=np.int32),
                    np.array([], dtype=np.int8),
                )
            paths = np.array([ref.img_path for ref in refs])
            rois = np.array([ref.roi_xyxy for ref in refs], dtype=np.int32)
            tile_ids = np.array([ref.tile_id for ref in refs], dtype=np.int8)
            return paths, rois, tile_ids

        clean_paths, clean_rois, clean_tile_ids = refs_to_arrays(self.clean_index)
        dirty_paths, dirty_rois, dirty_tile_ids = refs_to_arrays(self.dirty_index)

        try:
            np.savez_compressed(
                INDEX_CACHE_PATH,
                images_dir=np.array([self.images_dir]),
                labels_dir=np.array([self.labels_dir]),
                human_vehicle_classes=np.array(tuple(sorted(self.hv_classes)), dtype=np.int16),
                roi_pad_ratio=np.array([self.roi_pad_ratio], dtype=np.float32),
                min_pad_size=np.array([self.min_pad_size], dtype=np.int32),
                min_obj_intersection_ratio=np.array([self.min_obj_intersection_ratio], dtype=np.float32),
                clean_paths=clean_paths,
                clean_rois=clean_rois,
                clean_tile_ids=clean_tile_ids,
                dirty_paths=dirty_paths,
                dirty_rois=dirty_rois,
                dirty_tile_ids=dirty_tile_ids,
            )
            print(f"[INDEX] saved cache: {INDEX_CACHE_PATH}", flush=True)
        except Exception as exc:
            print(f"[INDEX] cache save skipped: {exc}", flush=True)

    def _build_index(self) -> None:
        if self._load_index_cache():
            return

        t0 = time.time()
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        img_files = [
            os.path.join(self.images_dir, f)
            for f in os.listdir(self.images_dir)
            if f.lower().endswith(exts)
        ]
        img_files.sort()

        counts = {
            "img_read_fail": 0,
            "empty_labels": 0,
            "pads_total": 0,
            "pads_small": 0,
            "pads_oob": 0,
            "clean_pads": 0,
            "dirty_pads": 0,
            "clean_tiles": 0,
            "dirty_tiles": 0,
        }

        for img_path in tqdm(img_files, desc="Indexing clean/dirty grid tiles", dynamic_ncols=True):
            base = os.path.splitext(os.path.basename(img_path))[0]
            labels = read_label_file(os.path.join(self.labels_dir, base + ".txt"))
            if not labels:
                counts["empty_labels"] += 1
                continue

            img = read_image(img_path)
            if img is None:
                counts["img_read_fail"] += 1
                continue
            img_h, img_w = img.shape[:2]

            normalized = coords_are_normalized(np.array(labels[0][1:5], dtype=np.float32))
            pads: List[Tuple[int, int, int, int]] = []
            objects: List[Tuple[int, int, int, int]] = []

            for cls, cx, cy, bw, bh in labels:
                box = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h, normalized)
                if cls in self.pad_classes:
                    pads.append(box)
                elif cls in self.hv_classes:
                    objects.append(clamp_xyxy(*box, img_w, img_h))

            for pad in pads:
                counts["pads_total"] += 1
                if not is_fully_visible(*pad, img_w, img_h):
                    counts["pads_oob"] += 1
                    continue

                pad_clamped = clamp_xyxy(*pad, img_w, img_h)
                if (pad_clamped[2] - pad_clamped[0]) < self.min_pad_size or (pad_clamped[3] - pad_clamped[1]) < self.min_pad_size:
                    counts["pads_small"] += 1
                    continue

                roi_xyxy = expand_box(pad_clamped, self.roi_pad_ratio, img_w, img_h)
                if not roi_xyxy:
                    counts["pads_oob"] += 1
                    continue

                is_dirty = has_human_or_vehicle(
                    pad_clamped,
                    objects,
                    min_intersection_ratio=self.min_obj_intersection_ratio,
                )
                if is_dirty:
                    ids = dirty_tile_ids(roi_xyxy, objects)
                    if ids:
                        refs = [TileRef(img_path, roi_xyxy, tile_id) for tile_id in ids]
                        self.dirty_index.extend(refs)
                        counts["dirty_pads"] += 1
                        counts["dirty_tiles"] += len(refs)
                else:
                    refs = [TileRef(img_path, roi_xyxy, tile_id) for tile_id in range(4)]
                    self.clean_index.extend(refs)
                    counts["clean_pads"] += 1
                    counts["clean_tiles"] += len(refs)

        print("\n[INDEX] DONE", flush=True)
        print(f"[INDEX] time          : {time.time() - t0:.1f}s", flush=True)
        for key, value in counts.items():
            print(f"[INDEX] {key:<14}: {value}", flush=True)
        print(f"[INDEX] clean index   : {len(self.clean_index)}", flush=True)
        print(f"[INDEX] dirty index   : {len(self.dirty_index)}", flush=True)

        if len(self.clean_index) < 100:
            raise RuntimeError("Not enough clean grid tiles. Check class ids, labels and filters.")
        if PENALTY_WEIGHT > 0.0 and len(self.dirty_index) < 10:
            raise RuntimeError("Not enough dirty grid tiles for hybrid penalty. Check human/vehicle labels.")
        self._save_index_cache()

    def __len__(self) -> int:
        return len(self.clean_index)

    def __getitem__(self, idx: int):
        clean_ref = self.clean_index[idx]
        dirty_ref = random.choice(self.dirty_index) if self.dirty_index else clean_ref

        clean_tile = load_tile(clean_ref)
        dirty_tile = load_tile(dirty_ref) if self.dirty_index else clean_tile

        v1 = self.transform(clean_tile)
        v2 = self.transform(clean_tile)
        v_unsafe = self.transform(dirty_tile)
        return v1, v2, v_unsafe


class SSLModel(nn.Module):
    def __init__(self, backbone_name: str = BACKBONE, proj_dim: int = PROJ_DIM) -> None:
        super().__init__()
        backbone_fn = getattr(torchvision.models, backbone_name, None)
        if backbone_fn is None:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        backbone = backbone_fn(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.projector(z)
        return nn.functional.normalize(z, dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    bsz = z1.shape[0]
    device_type = "cuda" if z1.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        z = torch.cat([z1.float(), z2.float()], dim=0)
        sim = torch.matmul(z, z.T) / temperature
        mask = torch.eye(2 * bsz, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
        pos = torch.cat([torch.diag(sim, bsz), torch.diag(sim, -bsz)], dim=0)
        return (-pos + torch.logsumexp(sim, dim=1)).mean()


def build_transform(img_size: int = IMG_SIZE) -> T.Compose:
    ops = [T.ToPILImage()]
    if USE_RANDOM_CROP:
        ops.append(T.RandomResizedCrop(img_size, scale=(0.90, 1.0), ratio=(0.90, 1.10)))
    else:
        ops.append(T.Resize((img_size, img_size)))

    ops.extend(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.01),
            T.RandomGrayscale(p=0.02),
        ]
    )
    if USE_GAUSSIAN_BLUR:
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.20))

    ops.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return T.Compose(ops)


def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_ssl_checkpoint(model: SSLModel, path: str, epoch: int) -> None:
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "projector": model.projector.state_dict(),
            "backbone": BACKBONE,
            "proj_dim": PROJ_DIM,
            "feature_space": "projector",
            "img_size": IMG_SIZE,
            "grid": "2x2",
            "human_vehicle_classes": HUMAN_VEHICLE_CLASSES,
            "roi_pad_ratio": ROI_PAD_RATIO,
            "min_pad_size": MIN_PAD_SIZE,
            "min_obj_intersection_ratio": MIN_OBJ_INTERSECTION_RATIO,
            "penalty_weight": PENALTY_WEIGHT,
            "unsafe_margin": UNSAFE_MARGIN,
            "epoch": epoch,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "use_amp": USE_AMP,
            "use_random_crop": USE_RANDOM_CROP,
            "use_gaussian_blur": USE_GAUSSIAN_BLUR,
        },
        path,
    )


def main() -> None:
    set_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    transform = build_transform(IMG_SIZE)

    dataset = HybridGridPadDataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        transform=transform,
        roi_pad_ratio=ROI_PAD_RATIO,
        min_pad_size=MIN_PAD_SIZE,
        min_obj_intersection_ratio=MIN_OBJ_INTERSECTION_RATIO,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device, flush=True)
    if device == "cuda":
        print("[CUDA]", torch.cuda.get_device_name(0), flush=True)

    use_amp = device == "cuda" and USE_AMP
    use_channels_last = device == "cuda" and USE_CHANNELS_LAST
    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": (device == "cuda"),
        "drop_last": True,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
    loader = DataLoader(**loader_kwargs)

    print(
        f"[CONFIG] backbone={BACKBONE} img_size={IMG_SIZE} batch_size={BATCH_SIZE} "
        f"epochs={EPOCHS} workers={NUM_WORKERS} amp={use_amp} channels_last={use_channels_last}",
        flush=True,
    )
    print(
        f"[CONFIG] steps_per_epoch={len(loader)} clean_tiles={len(dataset)} "
        f"random_crop={USE_RANDOM_CROP} gaussian_blur={USE_GAUSSIAN_BLUR}",
        flush=True,
    )

    model = SSLModel(backbone_name=BACKBONE, proj_dim=PROJ_DIM).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer_kwargs = {"lr": LR, "weight_decay": WEIGHT_DECAY}
    if device == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_device_type = "cuda" if device == "cuda" else "cpu"

    def move_to_device(x: torch.Tensor) -> torch.Tensor:
        if use_channels_last:
            return x.to(device, non_blocking=True, memory_format=torch.channels_last)
        return x.to(device, non_blocking=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        total_losses: List[float] = []
        simclr_losses: List[float] = []
        penalty_losses: List[float] = []

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False, dynamic_ncols=True)
        for v1, v2, v_unsafe in pbar:
            v1 = move_to_device(v1)
            v2 = move_to_device(v2)
            if PENALTY_WEIGHT > 0.0:
                v_unsafe = move_to_device(v_unsafe)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                z1 = model(v1)
                z2 = model(v2)

                loss_simclr = nt_xent_loss(z1, z2, TEMPERATURE)
                if PENALTY_WEIGHT > 0.0:
                    z_unsafe = model(v_unsafe)
                    unsafe_sim = torch.sum(z1 * z_unsafe, dim=1)
                    loss_penalty = torch.relu(unsafe_sim - UNSAFE_MARGIN).mean() * PENALTY_WEIGHT
                else:
                    loss_penalty = loss_simclr.new_zeros(())
                loss = loss_simclr + loss_penalty

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_losses.append(float(loss.item()))
            simclr_losses.append(float(loss_simclr.item()))
            penalty_losses.append(float(loss_penalty.item()))
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "simclr": f"{loss_simclr.item():.4f}",
                    "penalty": f"{loss_penalty.item():.4f}",
                }
            )

        epoch_time = time.time() - t0
        print(
            f"[SSL TRAIN] epoch={epoch:03d}/{EPOCHS} "
            f"loss={np.mean(total_losses):.4f} "
            f"simclr={np.mean(simclr_losses):.4f} "
            f"penalty={np.mean(penalty_losses):.4f} "
            f"time={epoch_time:.1f}s "
            f"steps_per_sec={len(loader) / max(epoch_time, 1e-9):.3f}",
            flush=True,
        )

        if SAVE_LATEST_EACH_EPOCH:
            save_ssl_checkpoint(model, LATEST_CKPT, epoch)
            print(f"[SSL TRAIN] latest checkpoint saved: {LATEST_CKPT}", flush=True)

    save_ssl_checkpoint(model, OUTPUT_CKPT, EPOCHS)
    print("Saved:", OUTPUT_CKPT, flush=True)


if __name__ == "__main__":
    main()
