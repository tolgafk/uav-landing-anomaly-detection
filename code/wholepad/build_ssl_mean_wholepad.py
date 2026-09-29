import os
from typing import List, Tuple
from tqdm import tqdm
import time
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

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


# Whole-pad memory-bank construction used for the paper comparison.

DATASET_DIR = os.environ.get("UAV_DATASET_DIR", r"path\to\2026_data_UAP_UAI")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SSL_CKPT = os.path.join(OUTPUT_DIR, "ssl_encoder_wholepad.pth")
OUT_BANK_MEMMAP = os.path.join(OUTPUT_DIR, "memory_bank_zero_margin.dat")
OUT_BANK_META = os.path.join(OUTPUT_DIR, "memory_bank_zero_margin_meta.npz")

PAD_CLASSES = (2, 3)
HUMAN_VEHICLE_CLASSES = ()
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
USE_AMP = True
USE_CHANNELS_LAST = True

# YENİ KURALLAR: %0 Margin
ROI_PAD_RATIO = 0.0
MIN_PAD_SIZE = 32

def _is_normalized(vals): return np.all(vals[1:] >= 0.0) and np.all(vals[1:] <= 1.0)
def yolo_to_xyxy(cx, cy, w, h, img_w, img_h, norm):
    if norm: cx *= img_w; w *= img_w; cy *= img_h; h *= img_h
    return int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))
def clamp_xyxy(x1, y1, x2, y2, w, h):
    return max(0, min(w - 1, x1)), max(0, min(h - 1, y1)), max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

def intersection_area(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1: return 0
    return (ix2 - ix1) * (iy2 - iy1)

class EncoderOnly(nn.Module):
    def __init__(self, backbone_name, state_dict):
        super().__init__()
        backbone = getattr(torchvision.models, backbone_name)(weights=None)
        backbone.fc = nn.Identity()
        backbone.load_state_dict(state_dict, strict=True)
        self.encoder = backbone
    def forward(self, x):
        return nn.functional.normalize(self.encoder(x), dim=1)

class CleanWholePadIndexDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir, self.labels_dir = images_dir, labels_dir
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.index = []
        self._build_index()

    def _build_index(self):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        img_files = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir) if f.lower().endswith(exts)]
        for img_path in tqdm(img_files, desc="Indexing MB (0% Margin)"):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lab_path = os.path.join(self.labels_dir, base + ".txt")
            if not os.path.exists(lab_path): continue
            
            with open(lab_path, "r") as f:
                labels = [list(map(float, line.strip().split())) for line in f if line.strip()]
            if not labels: continue
            
            img = read_image(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            norm = _is_normalized(np.array(labels[0][1:5]))

            pads, objects = [], []
            for lab in labels:
                x1, y1, x2, y2 = yolo_to_xyxy(lab[1], lab[2], lab[3], lab[4], w, h, norm)
                if int(lab[0]) in PAD_CLASSES: pads.append(clamp_xyxy(x1, y1, x2, y2, w, h))
                if int(lab[0]) in HUMAN_VEHICLE_CLASSES: objects.append(clamp_xyxy(x1, y1, x2, y2, w, h))

            for pad in pads:
                x1, y1, x2, y2 = pad
                if (x2 - x1) < MIN_PAD_SIZE or (y2 - y1) < MIN_PAD_SIZE: continue
                
                # 1 Piksel Kesişimi Bile Affetme
                is_dirty = any(intersection_area(pad, obj) > 0 for obj in objects)
                if not is_dirty:
                    self.index.append((img_path, pad))

        print(f"[INDEX-MB] Bankaya Eklenecek %0 Margin Temiz Pad Sayısı: {len(self.index)}")

    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        img_path, (x1, y1, x2, y2) = self.index[idx]
        img = read_image(img_path)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: return self.transform(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        return self.transform(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

def main():
    ckpt = torch.load(SSL_CKPT, map_location="cpu", weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = device == "cuda" and USE_AMP
    use_channels_last = device == "cuda" and USE_CHANNELS_LAST
    amp_device_type = "cuda" if device == "cuda" else "cpu"

    encoder = EncoderOnly(ckpt["backbone"], ckpt["encoder"]).to(device).eval()
    if use_channels_last:
        encoder = encoder.to(memory_format=torch.channels_last)

    dataset = CleanWholePadIndexDataset(IMAGES_DIR, LABELS_DIR)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": (device == "cuda"),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
    loader = DataLoader(**loader_kwargs)

    def move_to_device(x):
        if use_channels_last:
            return x.to(device, non_blocking=True, memory_format=torch.channels_last)
        return x.to(device, non_blocking=True)

    print(
        f"[CONFIG] batch_size={BATCH_SIZE} workers={NUM_WORKERS} "
        f"amp={use_amp} channels_last={use_channels_last}",
        flush=True,
    )

    with torch.inference_mode():
        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
            feat_dim = encoder(move_to_device(next(iter(loader)))).shape[1]
    
    bank = np.memmap(OUT_BANK_MEMMAP, mode="w+", dtype="float32", shape=(len(dataset), feat_dim))
    offset = 0
    with torch.inference_mode():
        for x in tqdm(loader, desc="Encoding Pads to Bank"):
            bsz = x.shape[0]
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                z = encoder(move_to_device(x))
            bank[offset:offset+bsz] = z.cpu().numpy().astype(np.float32)
            offset += bsz
    bank.flush()
    np.savez(OUT_BANK_META, n_tiles=len(dataset), feat_dim=feat_dim, dtype="float32")
    print(f"\n[BAŞARILI] {offset} Adet İniş Alanı Hafızaya Alındı.")

if __name__ == "__main__":
    main()
