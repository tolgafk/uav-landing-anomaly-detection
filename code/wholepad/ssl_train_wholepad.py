import os
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
import time
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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


# Whole-pad comparison pipeline used in the paper.

# --- DOSYA YOLLARI ---
DATASET_DIR = os.environ.get("UAV_DATASET_DIR", r"path\to\2026_data_UAP_UAI")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CKPT = os.path.join(OUTPUT_DIR, "ssl_encoder_wholepad.pth")

PAD_CLASSES = (2, 3)
HUMAN_VEHICLE_CLASSES = ()

BACKBONE = "resnet101"
IMG_SIZE = 224

BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4
TEMPERATURE = 0.2
PENALTY_WEIGHT = 0.0

# YENİ KURALLAR: %0 Margin 
ROI_PAD_RATIO = 0.0 
MIN_PAD_SIZE = 32

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
USE_AMP = True
USE_CHANNELS_LAST = True
USE_GAUSSIAN_BLUR = False
SEED = 42

def _is_normalized(vals: np.ndarray) -> bool: return np.all(vals[1:] >= 0.0) and np.all(vals[1:] <= 1.0)

def read_label_file(path: str) -> List[Tuple[int, float, float, float, float]]:
    if not os.path.exists(path): return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 5: continue
            out.append((int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return out

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h, normalized):
    if normalized: cx *= img_w; w *= img_w; cy *= img_h; h *= img_h
    return int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(cx + w / 2.0)), int(round(cy + h / 2.0))

def clamp_xyxy(x1, y1, x2, y2, img_w, img_h):
    return max(0, min(img_w - 1, x1)), max(0, min(img_h - 1, y1)), max(0, min(img_w - 1, x2)), max(0, min(img_h - 1, y2))

def intersection_area(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1: return 0
    return (ix2 - ix1) * (iy2 - iy1)

# --- DATASET ---
@dataclass
class PadRef:
    img_path: str
    roi_xyxy: Tuple[int, int, int, int]

class CleanWholePadDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform_clean, transform_tensor):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform_clean = transform_clean
        self.transform_tensor = transform_tensor
        
        # 2 Ayrı Havuz
        self.clean_index: List[PadRef] = []
        self.dirty_index: List[PadRef] = []
        
        self._build_index()

    def _build_index(self):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        img_files = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir) if f.lower().endswith(exts)]
        for img_path in tqdm(img_files, desc="Indexing Pads (Clean vs Dirty)"):
            base = os.path.splitext(os.path.basename(img_path))[0]
            labels = read_label_file(os.path.join(self.labels_dir, base + ".txt"))
            if not labels: continue
            
            img = read_image(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            norm = _is_normalized(np.array(labels[0][1:5], dtype=np.float32))

            pads, objects = [], []
            for cls, cx, cy, bw, bh in labels:
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h, norm)
                if cls in PAD_CLASSES: pads.append(clamp_xyxy(x1, y1, x2, y2, w, h))
                if cls in HUMAN_VEHICLE_CLASSES: objects.append(clamp_xyxy(x1, y1, x2, y2, w, h))

            for pad in pads:
                x1, y1, x2, y2 = pad
                if (x2 - x1) < MIN_PAD_SIZE or (y2 - y1) < MIN_PAD_SIZE: continue
                
                # İnsan veya araç kesişiyor mu?
                is_dirty = any(intersection_area(pad, obj) > 0 for obj in objects)
                
                if is_dirty:
                    self.dirty_index.append(PadRef(img_path, pad))
                else:
                    self.clean_index.append(PadRef(img_path, pad))

        print(f"[BİLGİ] {len(self.clean_index)} Adet TEMİZ İniş Alanı (SAFE) bulundu.")
        print(f"[BİLGİ] {len(self.dirty_index)} Adet KİRLİ İniş Alanı (UNSAFE - İnsan/Araçlı) bulundu.")
        
        if PENALTY_WEIGHT > 0.0 and len(self.dirty_index) == 0:
            raise ValueError("KİRLİ (İnsan/Araç olan) hiçbir pad bulunamadı! Cezalandırma için kirli veriye ihtiyaç var.")

    def __len__(self): return len(self.clean_index)
    
    def __getitem__(self, idx: int):
        # 1. TEMİZ PAD'İ HAZIRLA
        clean_ref = self.clean_index[idx]
        c_img = read_image(clean_ref.img_path)
        cx1, cy1, cx2, cy2 = clean_ref.roi_xyxy
        c_roi = c_img[cy1:cy2, cx1:cx2]
        if c_roi.size == 0: c_roi = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        c_pil = Image.fromarray(cv2.cvtColor(c_roi, cv2.COLOR_BGR2RGB))

        v1 = self.transform_tensor(self.transform_clean(c_pil))
        v2 = self.transform_tensor(self.transform_clean(c_pil))
        
        # 2. KİRLİ PAD'İ HAZIRLA (Rastgele seç)
        dirty_ref = random.choice(self.dirty_index) if self.dirty_index else clean_ref
        d_img = read_image(dirty_ref.img_path)
        dx1, dy1, dx2, dy2 = dirty_ref.roi_xyxy
        d_roi = d_img[dy1:dy2, dx1:dx2]
        if d_roi.size == 0: d_roi = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        d_pil = Image.fromarray(cv2.cvtColor(d_roi, cv2.COLOR_BGR2RGB))

        # Kirli pad'e de aynı hafif transformasyonları uygulayalım ki model augmentasyon farkını ezberlemesin
        v_unsafe = self.transform_tensor(self.transform_clean(d_pil))

        return v1, v2, v_unsafe

class SSLModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE, proj_dim=128):
        super().__init__()
        backbone = getattr(torchvision.models, backbone_name)(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, proj_dim))

    def forward(self, x):
        return nn.functional.normalize(self.projector(self.encoder(x)), dim=1)

def nt_xent_loss(z1, z2, temperature=0.2):
    bsz = z1.shape[0]
    device_type = "cuda" if z1.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        z = torch.cat([z1.float(), z2.float()], dim=0)
        sim = torch.matmul(z, z.T) / temperature
        mask = torch.eye(2 * bsz, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
        pos = torch.cat([torch.diag(sim, bsz), torch.diag(sim, -bsz)], dim=0)
        return (-pos + torch.logsumexp(sim, dim=1)).mean()

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    
    # Senin belirlediğin HAFİF (Mild) transformasyonlar
    transform_ops = [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5), 
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.01), 
        T.RandomGrayscale(p=0.02),
    ]
    if USE_GAUSSIAN_BLUR:
        transform_ops.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2))
    transform_clean = T.Compose(transform_ops)
    
    transform_tensor = T.Compose([
        T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    dataset = CleanWholePadDataset(IMAGES_DIR, LABELS_DIR, transform_clean, transform_tensor)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = device == "cuda" and USE_AMP
    use_channels_last = device == "cuda" and USE_CHANNELS_LAST
    amp_device_type = "cuda" if device == "cuda" else "cpu"

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": (device == "cuda"),
        "drop_last": True,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
    loader = DataLoader(**loader_kwargs)

    print(
        f"[CONFIG] backbone={BACKBONE} batch_size={BATCH_SIZE} epochs={EPOCHS} "
        f"workers={NUM_WORKERS} amp={use_amp} channels_last={use_channels_last}",
        flush=True,
    )

    model = SSLModel(backbone_name=BACKBONE).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer_kwargs = {"lr": LR}
    if device == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except TypeError:
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def move_to_device(x):
        if use_channels_last:
            return x.to(device, non_blocking=True, memory_format=torch.channels_last)
        return x.to(device, non_blocking=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for v1, v2, v_unsafe in pbar:
            v1, v2 = move_to_device(v1), move_to_device(v2)
            if PENALTY_WEIGHT > 0.0:
                v_unsafe = move_to_device(v_unsafe)
            
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                z1, z2 = model(v1), model(v2)
            
            # 1. SimCLR Loss: Temiz Pad'leri (Gölge farklarına rağmen) birbirine yakınlaştır
            loss_simclr = nt_xent_loss(z1, z2, TEMPERATURE)
            
            # 2. GERÇEK VERİ İLE CEZALANDIRMA: Temiz Pad ile Gerçekten Kirli (İnsanlı) Pad'i birbirinden UZAKLAŞTIR!
            if PENALTY_WEIGHT > 0.0:
                z_unsafe = model(v_unsafe)
                sim_unsafe = torch.sum(z1.float() * z_unsafe.float(), dim=1)
                loss_penalty = torch.relu(sim_unsafe).mean() * PENALTY_WEIGHT
            else:
                loss_penalty = loss_simclr.new_zeros(())
            
            total_loss = loss_simclr + loss_penalty

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            losses.append(total_loss.item())
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "penalty": f"{loss_penalty.item():.4f}"})
            
        print(f"[SSL TRAIN] Epoch {epoch:03d} | Total Loss = {np.mean(losses):.4f}")

    torch.save({"encoder": model.encoder.state_dict(), "backbone": BACKBONE}, OUTPUT_CKPT)
    print("Gerçek Veriyle Cezalandırılmış Model Kaydedildi:", OUTPUT_CKPT)

if __name__ == "__main__":
    main()
