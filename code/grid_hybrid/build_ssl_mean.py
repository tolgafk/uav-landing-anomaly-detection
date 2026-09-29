import os
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


DATASET_DIR = os.environ.get("UAV_DATASET_DIR", r"path\to\2026_data_UAP_UAI")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SSL_CKPT = os.path.join(OUTPUT_DIR, "ssl_encoder_grid_hybrid.pth")
OUT_BANK_MEMMAP = os.path.join(OUTPUT_DIR, "memory_bank_grid_hybrid.dat")
OUT_BANK_META = os.path.join(OUTPUT_DIR, "memory_bank_grid_hybrid_meta.npz")

PAD_CLASSES = (2, 3)
HUMAN_VEHICLE_CLASSES = ()

IMG_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
USE_AMP = True
USE_CHANNELS_LAST = True

ROI_PAD_RATIO = 0.0
MIN_PAD_SIZE = 32
MIN_OBJ_INTERSECTION_RATIO = 0.0


@dataclass(frozen=True)
class TileRef:
    img_path: str
    roi_xyxy: Tuple[int, int, int, int]
    tile_id: int


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


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int, normalized: bool):
    if normalized:
        cx *= img_w
        w *= img_w
        cy *= img_h
        h *= img_h
    return (
        int(round(cx - w / 2.0)),
        int(round(cy - h / 2.0)),
        int(round(cx + w / 2.0)),
        int(round(cy + h / 2.0)),
    )


def clamp_xyxy(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int):
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
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def has_human_or_vehicle(pad_xyxy, objects_xyxy, min_intersection_ratio: float) -> bool:
    for obj in objects_xyxy:
        if intersection_area(pad_xyxy, obj) > 0:
            return True
    return False


def expand_box(box, ratio: float, img_w: int, img_h: int):
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


class ProjectionEncoder(nn.Module):
    def __init__(self, ckpt: dict) -> None:
        super().__init__()
        backbone_name = ckpt.get("backbone", "resnet101")
        backbone = getattr(torchvision.models, backbone_name)(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        backbone.load_state_dict(ckpt["encoder"], strict=True)
        self.encoder = backbone

        self.projector = None
        if "projector" in ckpt:
            proj_dim = int(ckpt.get("proj_dim", 128))
            projector = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, proj_dim),
            )
            projector.load_state_dict(ckpt["projector"], strict=True)
            self.projector = projector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.projector is not None:
            z = self.projector(z)
        return nn.functional.normalize(z, dim=1)


class CleanGridTileDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        roi_pad_ratio: float,
        min_pad_size: int,
        min_obj_intersection_ratio: float,
        img_size: int,
    ) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.roi_pad_ratio = roi_pad_ratio
        self.min_pad_size = min_pad_size
        self.min_obj_intersection_ratio = min_obj_intersection_ratio
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.index: List[TileRef] = []
        self._build_index()

    def _build_index(self) -> None:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        img_files = [
            os.path.join(self.images_dir, f)
            for f in os.listdir(self.images_dir)
            if f.lower().endswith(exts)
        ]
        img_files.sort()

        counts = {
            "pads_total": 0,
            "pads_clean": 0,
            "pads_dirty_skip": 0,
            "pads_small_skip": 0,
            "pads_oob_skip": 0,
            "tiles": 0,
        }

        for img_path in tqdm(img_files, desc="Indexing clean grid bank", dynamic_ncols=True):
            base = os.path.splitext(os.path.basename(img_path))[0]
            labels = read_label_file(os.path.join(self.labels_dir, base + ".txt"))
            if not labels:
                continue

            img = read_image(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            normalized = coords_are_normalized(np.array(labels[0][1:5], dtype=np.float32))

            pads = []
            objects = []
            for cls, cx, cy, bw, bh in labels:
                box = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h, normalized)
                if cls in PAD_CLASSES:
                    pads.append(box)
                elif cls in HUMAN_VEHICLE_CLASSES:
                    objects.append(clamp_xyxy(*box, img_w, img_h))

            for pad in pads:
                counts["pads_total"] += 1
                if not is_fully_visible(*pad, img_w, img_h):
                    counts["pads_oob_skip"] += 1
                    continue
                pad_clamped = clamp_xyxy(*pad, img_w, img_h)
                if (pad_clamped[2] - pad_clamped[0]) < self.min_pad_size or (pad_clamped[3] - pad_clamped[1]) < self.min_pad_size:
                    counts["pads_small_skip"] += 1
                    continue
                if has_human_or_vehicle(pad_clamped, objects, self.min_obj_intersection_ratio):
                    counts["pads_dirty_skip"] += 1
                    continue

                roi_xyxy = expand_box(pad_clamped, self.roi_pad_ratio, img_w, img_h)
                if not roi_xyxy:
                    counts["pads_oob_skip"] += 1
                    continue

                for tile_id in range(4):
                    self.index.append(TileRef(img_path, roi_xyxy, tile_id))
                    counts["tiles"] += 1
                counts["pads_clean"] += 1

        print("\n[INDEX-MB] DONE", flush=True)
        for key, value in counts.items():
            print(f"[INDEX-MB] {key:<16}: {value}", flush=True)
        print(f"[INDEX-MB] final tiles     : {len(self.index)}", flush=True)
        if not self.index:
            raise RuntimeError("No clean tiles found for memory bank.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ref = self.index[idx]
        img = read_image(ref.img_path)
        if img is None:
            return self.transform(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

        x1, y1, x2, y2 = ref.roi_xyxy
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return self.transform(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

        tiles = split_grid_2x2(roi)
        if not tiles:
            return self.transform(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

        tile = tiles[min(ref.tile_id, len(tiles) - 1)]
        rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        return self.transform(rgb)


def main() -> None:
    if not os.path.exists(SSL_CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {SSL_CKPT}. Run ssl_train_grid.py first.")

    ckpt = torch.load(SSL_CKPT, map_location="cpu")
    roi_pad_ratio = float(ckpt.get("roi_pad_ratio", ROI_PAD_RATIO))
    min_pad_size = int(ckpt.get("min_pad_size", MIN_PAD_SIZE))
    min_obj_intersection_ratio = float(ckpt.get("min_obj_intersection_ratio", MIN_OBJ_INTERSECTION_RATIO))
    img_size = int(ckpt.get("img_size", IMG_SIZE))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device, flush=True)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = device == "cuda" and USE_AMP
    use_channels_last = device == "cuda" and USE_CHANNELS_LAST
    amp_device_type = "cuda" if device == "cuda" else "cpu"

    model = ProjectionEncoder(ckpt).to(device).eval()
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    dataset = CleanGridTileDataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        roi_pad_ratio=roi_pad_ratio,
        min_pad_size=min_pad_size,
        min_obj_intersection_ratio=min_obj_intersection_ratio,
        img_size=img_size,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": (device == "cuda"),
        "drop_last": False,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
    loader = DataLoader(**loader_kwargs)

    print(
        f"[CONFIG] batch_size={BATCH_SIZE} workers={NUM_WORKERS} "
        f"amp={use_amp} channels_last={use_channels_last}",
        flush=True,
    )

    def move_to_device(x: torch.Tensor) -> torch.Tensor:
        if use_channels_last:
            return x.to(device, non_blocking=True, memory_format=torch.channels_last)
        return x.to(device, non_blocking=True)

    with torch.inference_mode():
        first_batch = move_to_device(next(iter(loader)))
        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            feat_dim = model(first_batch).shape[1]

    n_tiles = len(dataset)
    print(f"[BANK] tiles={n_tiles} feat_dim={feat_dim}", flush=True)
    bank = np.memmap(OUT_BANK_MEMMAP, mode="w+", dtype="float32", shape=(n_tiles, feat_dim))

    offset = 0
    with torch.inference_mode():
        for x in tqdm(loader, desc="Encoding clean grid tiles", dynamic_ncols=True):
            bsz = x.shape[0]
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                z = model(move_to_device(x))
            z = z.detach().cpu().numpy().astype(np.float32)
            bank[offset : offset + bsz] = z
            offset += bsz
    bank.flush()

    np.savez(
        OUT_BANK_META,
        n_tiles=n_tiles,
        feat_dim=feat_dim,
        dtype="float32",
        checkpoint=SSL_CKPT,
        feature_space=ckpt.get("feature_space", "encoder"),
        grid="2x2",
        roi_pad_ratio=roi_pad_ratio,
        min_obj_intersection_ratio=min_obj_intersection_ratio,
    )
    print("Saved memmap:", OUT_BANK_MEMMAP, flush=True)
    print("Saved meta  :", OUT_BANK_META, flush=True)


if __name__ == "__main__":
    main()
