import os
import shutil
import cv2
import random
import numpy as np
from tqdm import tqdm


def read_image(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str, img) -> bool:
    ext = os.path.splitext(path)[1]
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    encoded.tofile(path)
    return True


# ==============================
# KLASÖR AYARLARI
# ==============================
BASE_DIR = os.environ.get("UAV_DATASET_DIR", r"path\to\2026_data_UAP_UAI")

# Kaynak (Orijinal Test Seti)
IMG_DIR_SRC = os.path.join(BASE_DIR, "images", "test")
LAB_DIR_SRC = os.path.join(BASE_DIR, "labels", "test")

# Hedef (Yeni Oluşturulacak Test Seti - SADECE 30 GÖRÜNTÜ OLACAK)
IMG_DIR_DST = os.path.join(BASE_DIR, "images", "test_augmented")
LAB_DIR_DST = os.path.join(BASE_DIR, "labels", "test_augmented")

# Kaç görüntüye müdahale edilecek?
NUM_AUGMENT_IMAGES = 30
PAD_CLASSES = {2, 3}

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def add_occlusion_to_pad(img, x1, y1, x2, y2):
    pad_w = x2 - x1
    pad_h = y2 - y1
    
    # Çok küçük pad'leri atla
    if pad_w < 15 or pad_h < 15:
        return img

    pad_area = pad_w * pad_h
    
    # %5 ile %50 arasında rastgele bir kaplama (occlusion) alanı seç
    coverage_ratio = random.uniform(0.05, 0.50)
    target_area = pad_area * coverage_ratio

    # Cisim için rastgele bir en/boy oranı (aspect ratio) belirle
    aspect_ratio = random.uniform(0.5, 2.0)
    obj_w = int(np.sqrt(target_area * aspect_ratio))
    obj_h = int(np.sqrt(target_area / aspect_ratio))

    # Cismin boyutlarının, pad'in sınırlarını aşmadığından emin ol
    obj_w = min(obj_w, pad_w - 1)
    obj_h = min(obj_h, pad_h - 1)

    if obj_w < 2 or obj_h < 2:
        return img

    # Cismi pad'in içerisinde rastgele bir konuma yerleştir
    ox = random.randint(x1, x1 + pad_w - obj_w)
    oy = random.randint(y1, y1 + pad_h - obj_h)

    # Renk Paleti (BGR formatında)
    colors = [
        (0, 0, 0),         # Saf Siyah
        (40, 40, 40),      # Koyu Gri (Lastik / Kutu)
        (30, 30, 180),     # Kırmızımsı Kutu
        (150, 50, 30),     # Mavimsi Kutu
        (20, 100, 20)      # Koyu Yeşil
    ]
    color = random.choice(colors)
    shape_type = random.choice(["rectangle", "ellipse"])

    if shape_type == "rectangle":
        cv2.rectangle(img, (ox, oy), (ox + obj_w, oy + obj_h), color, thickness=-1)
    else:
        center = (ox + obj_w // 2, oy + obj_h // 2)
        axes = (obj_w // 2, obj_h // 2)
        cv2.ellipse(img, center, axes, 0, 0, 360, color, thickness=-1)

    return img

def main():
    # Yeni klasörleri oluştur (Varsa içini temizle)
    for d in [IMG_DIR_DST, LAB_DIR_DST]:
        os.makedirs(d, exist_ok=True)

    # İlk 30 dosyayı doğrudan alıyoruz
    img_files = [f for f in os.listdir(IMG_DIR_SRC) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:NUM_AUGMENT_IMAGES]
    
    print(f"[BİLGİ] {len(img_files)} test görüntüsü seçildi. Sadece bunlar işlenip yeni klasöre kaydedilecek...")

    for fname in tqdm(img_files, desc="Augmentasyon Uygulanıyor"):
        img_src_path = os.path.join(IMG_DIR_SRC, fname)
        img_dst_path = os.path.join(IMG_DIR_DST, fname)
        
        base_name = os.path.splitext(fname)[0]
        lab_src_path = os.path.join(LAB_DIR_SRC, base_name + ".txt")
        lab_dst_path = os.path.join(LAB_DIR_DST, base_name + ".txt")

        # 1. Label dosyasını yeni klasöre kopyala
        if os.path.exists(lab_src_path):
            shutil.copy(lab_src_path, lab_dst_path)

        # 2. Görüntüyü oku ve müdahale et
        img = read_image(img_src_path)
        if img is None: continue
        
        h, w = img.shape[:2]
        
        # Etiketleri oku ve Pad'leri bul
        if os.path.exists(lab_src_path):
            with open(lab_src_path, "r") as f:
                lines = [line.strip().split() for line in f if line.strip()]
            
            for parts in lines:
                cls, cx, cy, bw, bh = map(float, parts[:5])
                if int(cls) in PAD_CLASSES:
                    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
                    img = add_occlusion_to_pad(img, x1, y1, x2, y2)
        
        # Sadece müdahale edilmiş bu görüntüyü kaydet
        write_image(img_dst_path, img)

    print(f"\n[BAŞARILI] İşlem tamamlandı! Sadece 30 görüntülük özel test setin şurada hazır:")
    print(f"Görüntüler: {IMG_DIR_DST}")
    print(f"Etiketler:  {LAB_DIR_DST}")

if __name__ == "__main__":
    main()
