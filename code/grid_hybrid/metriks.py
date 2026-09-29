import json
import os
import random
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tqdm import tqdm


def read_image(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


DATASET_DIR = os.environ.get("UAV_DATASET_DIR", r"path\to\2026_data_UAP_UAI")
IMAGE_DIR = os.path.join(DATASET_DIR, "images", "test")
LABEL_DIR = os.path.join(DATASET_DIR, "labels", "test")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

SSL_WEIGHTS = os.path.join(OUTPUT_DIR, "ssl_encoder_grid_hybrid.pth")
BANK_META = os.path.join(OUTPUT_DIR, "memory_bank_grid_hybrid_meta.npz")
BANK_DAT = os.path.join(OUTPUT_DIR, "memory_bank_grid_hybrid.dat")

OUT_DIR = os.path.join(SCRIPT_DIR, "metriks")
OUT_JSON = os.path.join(OUT_DIR, "ssl_synthetic_metrics_final.json")
OUT_ROC = os.path.join(OUT_DIR, "ssl_synthetic_roc_final.png")
OUT_HIST = os.path.join(OUT_DIR, "ssl_synthetic_hist_final.png")
OUT_CM = os.path.join(OUT_DIR, "ssl_synthetic_confusion_final.png")
OUT_THRESHOLD = os.path.join(OUTPUT_DIR, "anomaly_threshold.json")

IMG_SIZE = 224
PAD_CLASSES = {2, 3}
HUMAN_VEHICLE_CLASSES = set()
MIN_OBJ_INTERSECTION_RATIO = 0.0
MIN_PAD_SIZE = 32
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def coords_are_normalized(vals: np.ndarray) -> bool:
    return np.all(vals >= 0.0) and np.all(vals <= 1.0)


def read_yolo(path: str) -> List[Tuple[int, float, float, float, float]]:
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
    if ex1 < 0 or ey1 < 0 or ex2 > img_w or ey2 > img_h or ex2 <= ex1 or ey2 <= ey1:
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


def choose_contrasting_color(patch_rgb: np.ndarray) -> Tuple[int, int, int]:
    median_color = np.median(patch_rgb.reshape(-1, 3), axis=0)
    palette = np.array(
        [
            (15, 15, 15),
            (235, 235, 235),
            (255, 170, 0),
            (210, 35, 35),
            (25, 90, 230),
            (30, 180, 65),
            (180, 35, 210),
            (0, 210, 220),
        ],
        dtype=np.float32,
    )
    distances = np.linalg.norm(palette - median_color.astype(np.float32), axis=1)
    best_indices = np.argsort(distances)[-4:]
    color = palette[int(random.choice(best_indices))].astype(np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def make_obstacle_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)

    coverage_ratio = random.uniform(0.22, 0.65)
    target_area = h * w * coverage_ratio
    shape_type = random.choice(["rotated_rect", "ellipse", "band", "polygon"])

    if shape_type == "band":
        obj_w = random.randint(max(4, int(w * 0.55)), max(4, w - 1))
        obj_h = random.randint(max(4, int(h * 0.18)), max(4, int(h * 0.42)))
    else:
        aspect_ratio = random.uniform(0.45, 2.6)
        obj_w = min(max(4, int(np.sqrt(target_area * aspect_ratio))), max(4, w - 1))
        obj_h = min(max(4, int(np.sqrt(target_area / aspect_ratio))), max(4, h - 1))

    cx = random.randint(obj_w // 2, max(obj_w // 2, w - obj_w // 2 - 1))
    cy = random.randint(obj_h // 2, max(obj_h // 2, h - obj_h // 2 - 1))
    angle = random.uniform(-55.0, 55.0)

    if shape_type in {"rotated_rect", "band"}:
        rect = ((float(cx), float(cy)), (float(obj_w), float(obj_h)), float(angle))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [box], 255)
    elif shape_type == "ellipse":
        axes = (max(2, obj_w // 2), max(2, obj_h // 2))
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, thickness=-1)
    else:
        points = []
        radius_x = max(3, obj_w // 2)
        radius_y = max(3, obj_h // 2)
        for i in range(8):
            theta = (2.0 * np.pi * i / 8.0) + random.uniform(-0.25, 0.25)
            scale = random.uniform(0.65, 1.15)
            px = int(cx + np.cos(theta) * radius_x * scale)
            py = int(cy + np.sin(theta) * radius_y * scale)
            points.append((max(0, min(w - 1, px)), max(0, min(h - 1, py))))
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)

    return mask


def add_shadow(img: np.ndarray, mask: np.ndarray) -> None:
    h, w = mask.shape
    offset = max(2, int(min(h, w) * 0.035))
    matrix = np.float32([[1, 0, offset], [0, 1, offset]])
    shadow_mask = cv2.warpAffine(mask, matrix, (w, h), borderValue=0) > 0
    object_mask = mask > 0
    shadow_only = shadow_mask & ~object_mask
    img[shadow_only] = (img[shadow_only].astype(np.float32) * 0.55).astype(np.uint8)


def add_object_texture(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> None:
    object_mask = mask > 0
    img[object_mask] = np.array(color, dtype=np.uint8)

    brightness = sum(color) / 3.0
    stripe_color = (20, 20, 20) if brightness > 130 else (240, 240, 240)
    stripe_layer = img.copy()
    h, w = mask.shape
    spacing = max(10, min(h, w) // 5)
    thickness = max(2, min(h, w) // 38)

    for x in range(-h, w + h, spacing):
        cv2.line(stripe_layer, (x, 0), (x + h, h), stripe_color, thickness, cv2.LINE_AA)

    img[object_mask] = stripe_layer[object_mask]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_color = (0, 0, 0) if brightness > 130 else (255, 255, 255)
    cv2.drawContours(img, contours, -1, edge_color, max(2, min(h, w) // 55), cv2.LINE_AA)


def add_synthetic_occlusion(patch_rgb: np.ndarray) -> np.ndarray:
    img = patch_rgb.copy()
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return img

    mask = make_obstacle_mask(h, w)
    color = choose_contrasting_color(patch_rgb)
    add_shadow(img, mask)
    add_object_texture(img, mask, color)

    return img


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
        return F.normalize(z, dim=1)


@torch.no_grad()
def score_roi_grid(
    roi_rgb: np.ndarray,
    model: ProjectionEncoder,
    transform,
    bank_tensor: torch.Tensor,
    device: str,
) -> float:
    tiles = split_grid_2x2(roi_rgb)
    if not tiles:
        return 1.0

    batch = torch.stack([transform(tile) for tile in tiles], dim=0).to(device)
    z = model(batch)
    sim = z @ bank_tensor.T
    tile_scores = 1.0 - sim.max(dim=1).values
    return float(tile_scores.max().item())


def collect_clean_pad_rois(image_dir: str, label_dir: str, roi_pad_ratio: float):
    rois = []
    stats = {
        "images": 0,
        "pads_total": 0,
        "pads_clean": 0,
        "pads_hv_skip": 0,
        "pads_small_skip": 0,
        "pads_oob_skip": 0,
    }

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_names = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(exts)])
    for fname in tqdm(image_names, desc="Collecting clean test pads", dynamic_ncols=True):
        img_path = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, os.path.splitext(fname)[0] + ".txt")
        labels = read_yolo(label_path)
        if not labels:
            continue

        img = read_image(img_path)
        if img is None:
            continue
        stats["images"] += 1
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
            stats["pads_total"] += 1
            pad = clamp_xyxy(*pad, img_w, img_h)
            if (pad[2] - pad[0]) < MIN_PAD_SIZE or (pad[3] - pad[1]) < MIN_PAD_SIZE:
                stats["pads_small_skip"] += 1
                continue
            if has_human_or_vehicle(pad, objects, MIN_OBJ_INTERSECTION_RATIO):
                stats["pads_hv_skip"] += 1
                continue

            roi_xyxy = expand_box(pad, roi_pad_ratio, img_w, img_h)
            if not roi_xyxy:
                stats["pads_oob_skip"] += 1
                continue

            x1, y1, x2, y2 = roi_xyxy
            roi_bgr = img[y1:y2, x1:x2]
            if roi_bgr.size == 0:
                stats["pads_oob_skip"] += 1
                continue

            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            rois.append(roi_rgb)
            stats["pads_clean"] += 1

    return rois, stats


def choose_best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return float(np.median(scores))
    f1 = (2.0 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def load_decision_threshold(path: str, fallback_threshold: float) -> Tuple[float, str]:
    if not os.path.exists(path):
        return float(fallback_threshold), "fallback_youden_j_statistic"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data["threshold"]), "json_threshold"
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"[WARN] Could not read threshold from {path}: {exc}")
        return float(fallback_threshold), "fallback_youden_j_statistic"


def evaluate_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(tn + fp, 1)
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def save_plots(scores: np.ndarray, labels: np.ndarray, fpr, tpr, auroc: float, threshold: float, cm: np.ndarray) -> None:
    plt.figure()
    plt.hist(scores[labels == 0], bins=40, alpha=0.7, label="SAFE")
    plt.hist(scores[labels == 1], bins=40, alpha=0.7, label="UNSAFE")
    plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label=f"Thr: {threshold:.4f}")
    plt.legend()
    plt.xlabel("Anomaly score = max tile(1 - max cosine similarity)")
    plt.ylabel("Count")
    plt.title("Grid-Hybrid SSL Score Distribution")
    plt.tight_layout()
    plt.savefig(OUT_HIST, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Grid-Hybrid SSL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ROC, dpi=200)
    plt.close()

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["SAFE", "UNSAFE"])
    plt.yticks([0, 1], ["SAFE", "UNSAFE"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_CM, dpi=200)
    plt.close()


def main() -> None:
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(SSL_WEIGHTS):
        raise FileNotFoundError(f"Missing checkpoint: {SSL_WEIGHTS}. Run ssl_train_grid.py first.")
    if not os.path.exists(BANK_META) or not os.path.exists(BANK_DAT):
        raise FileNotFoundError("Missing memory bank. Run build_ssl_mean.py after training.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device, flush=True)

    ckpt = torch.load(SSL_WEIGHTS, map_location="cpu")
    roi_pad_ratio = float(ckpt.get("roi_pad_ratio", 0.0))
    model = ProjectionEncoder(ckpt).to(device).eval()

    meta = np.load(BANK_META)
    n_tiles = int(meta["n_tiles"])
    feat_dim = int(meta["feat_dim"])
    bank_mem = np.memmap(BANK_DAT, mode="r", dtype=np.float32, shape=(n_tiles, feat_dim))
    bank_tensor = torch.from_numpy(np.array(bank_mem)).to(device)
    bank_tensor = F.normalize(bank_tensor, dim=1)

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    rois, test_stats = collect_clean_pad_rois(IMAGE_DIR, LABEL_DIR, roi_pad_ratio)
    if not rois:
        raise RuntimeError("No clean test pads found for synthetic evaluation.")

    scores = []
    labels = []
    with torch.no_grad():
        for roi_rgb in tqdm(rois, desc="Scoring clean/synthetic pads", dynamic_ncols=True):
            safe_score = score_roi_grid(roi_rgb, model, transform, bank_tensor, device)
            scores.append(safe_score)
            labels.append(0)

            unsafe_roi = add_synthetic_occlusion(roi_rgb)
            unsafe_score = score_roi_grid(unsafe_roi, model, transform, bank_tensor, device)
            scores.append(unsafe_score)
            labels.append(1)

    scores_np = np.array(scores, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int64)

    auroc = roc_auc_score(labels_np, scores_np)
    fpr, tpr, roc_thresholds = roc_curve(labels_np, scores_np)
    youden_threshold = float(roc_thresholds[int(np.argmax(tpr - fpr))])
    best_f1_threshold = choose_best_f1_threshold(labels_np, scores_np)
    decision_threshold, threshold_selection = load_decision_threshold(OUT_THRESHOLD, youden_threshold)
    decision_eval = evaluate_threshold(labels_np, scores_np, decision_threshold)
    youden_eval = evaluate_threshold(labels_np, scores_np, youden_threshold)
    best_f1_eval = evaluate_threshold(labels_np, scores_np, best_f1_threshold)
    cm_values = decision_eval["confusion_matrix"]
    cm = np.array(
        [
            [cm_values["tn"], cm_values["fp"]],
            [cm_values["fn"], cm_values["tp"]],
        ],
        dtype=np.int64,
    )

    metrics = {
        "dataset": "TEST clean pads + synthetic occlusion anomalies",
        "pipeline": "grid_hybrid_ssl",
        "checkpoint": os.path.basename(SSL_WEIGHTS),
        "memory_bank": os.path.basename(BANK_DAT),
        "feature_space": ckpt.get("feature_space", "encoder"),
        "grid": "2x2",
        "roi_pad_ratio": roi_pad_ratio,
        "total_samples": int(len(scores_np)),
        "safe_samples": int((labels_np == 0).sum()),
        "unsafe_samples": int((labels_np == 1).sum()),
        "test_pad_stats": test_stats,
        "AUROC": float(auroc),
        "decision_threshold": float(decision_threshold),
        "youden_threshold": float(youden_threshold),
        "best_f1_threshold": float(best_f1_threshold),
        "optimal_threshold": float(decision_threshold),
        "threshold_selection": threshold_selection,
        "precision": decision_eval["precision"],
        "recall": decision_eval["recall"],
        "specificity": decision_eval["specificity"],
        "f1_score": decision_eval["f1_score"],
        "accuracy": decision_eval["accuracy"],
        "balanced_accuracy": decision_eval["balanced_accuracy"],
        "confusion_matrix": decision_eval["confusion_matrix"],
        "threshold_metrics": {
            threshold_selection: decision_eval,
            "youden_j_statistic": youden_eval,
            "best_f1_on_synthetic_eval": best_f1_eval,
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    with open(OUT_THRESHOLD, "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(decision_threshold),
                "youden_threshold": float(youden_threshold),
                "best_f1_threshold": float(best_f1_threshold),
                "threshold_selection": threshold_selection,
                "model": os.path.basename(SSL_WEIGHTS),
                "bank_memmap": os.path.basename(BANK_DAT),
                "bank_meta": os.path.basename(BANK_META),
                "grid": "2x2",
                "roi_pad_ratio": roi_pad_ratio,
                "min_obj_intersection_ratio": MIN_OBJ_INTERSECTION_RATIO,
                "score": "max_tile_1_minus_max_cosine",
            },
            f,
            indent=4,
        )

    save_plots(scores_np, labels_np, fpr, tpr, auroc, decision_threshold, cm)

    print("\n--- FINAL GRID-HYBRID SSL RESULTS ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.upper():<24}: {value:.4f}")
        elif key not in {"test_pad_stats", "confusion_matrix", "threshold_metrics"}:
            print(f"{key.upper():<24}: {value}")
    print("TEST_PAD_STATS          :", test_stats)
    print("CONFUSION_MATRIX        :", metrics["confusion_matrix"])
    print("THRESHOLD_COMPARISON    :")
    for name, values in metrics["threshold_metrics"].items():
        print(
            f"  {name:<26} thr={values['threshold']:.4f} "
            f"p={values['precision']:.4f} r={values['recall']:.4f} "
            f"spec={values['specificity']:.4f} f1={values['f1_score']:.4f}"
        )
    print("Saved results to        :", OUT_DIR)


if __name__ == "__main__":
    main()
