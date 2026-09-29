import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import metriks


DATASET_DIR = Path(os.environ.get("UAV_DATASET_DIR", r"path\to\2026_data_UAP_UAI"))
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_IMAGE_DIR = DATASET_DIR / "images" / "test"
DEFAULT_LABEL_DIR = DATASET_DIR / "labels" / "test"
DEFAULT_CKPT = OUTPUT_DIR / "ssl_encoder_grid_hybrid.pth"
DEFAULT_BANK_META = OUTPUT_DIR / "memory_bank_grid_hybrid_meta.npz"
DEFAULT_BANK_DAT = OUTPUT_DIR / "memory_bank_grid_hybrid.dat"
DEFAULT_THRESHOLD_JSON = OUTPUT_DIR / "anomaly_threshold.json"

PAD_CLASSES = {2, 3}
HUMAN_VEHICLE_CLASSES = {0, 1}
WINDOW_NAME = "Grid-Hybrid SSL Test ROI Viewer"


def load_threshold(path: Path, mode: str, manual_threshold: float | None) -> float:
    if manual_threshold is not None:
        return float(manual_threshold)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if mode == "youden":
        return float(data.get("youden_threshold", data["threshold"]))
    return float(data["threshold"])


def load_bank(meta_path: Path, dat_path: Path, device: str) -> torch.Tensor:
    meta = np.load(meta_path)
    n_tiles = int(meta["n_tiles"])
    feat_dim = int(meta["feat_dim"])
    bank_mem = np.memmap(dat_path, mode="r", dtype=np.float32, shape=(n_tiles, feat_dim))
    bank_tensor = torch.from_numpy(np.array(bank_mem)).to(device)
    return F.normalize(bank_tensor, dim=1)


def collect_pad_boxes(label_path: Path, img_w: int, img_h: int) -> Tuple[List[Tuple[int, Tuple[int, int, int, int]]], List[Tuple[int, Tuple[int, int, int, int]]]]:
    labels = metriks.read_yolo(str(label_path))
    if not labels:
        return [], []

    normalized = metriks.coords_are_normalized(np.array(labels[0][1:5], dtype=np.float32))
    pads = []
    objects = []
    for cls, cx, cy, bw, bh in labels:
        box = metriks.yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h, normalized)
        box = metriks.clamp_xyxy(*box, img_w, img_h)
        if cls in PAD_CLASSES:
            pads.append((cls, box))
        elif cls in HUMAN_VEHICLE_CLASSES:
            objects.append((cls, box))
    return pads, objects


def score_pad(
    img_bgr: np.ndarray,
    pad_box: Tuple[int, int, int, int],
    roi_pad_ratio: float,
    model: metriks.ProjectionEncoder,
    transform,
    bank_tensor: torch.Tensor,
    device: str,
) -> Tuple[float | None, Tuple[int, int, int, int] | None]:
    img_h, img_w = img_bgr.shape[:2]
    roi_xyxy = metriks.expand_box(pad_box, roi_pad_ratio, img_w, img_h)
    if not roi_xyxy:
        return None, None

    x1, y1, x2, y2 = roi_xyxy
    roi_bgr = img_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None, None

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    score = metriks.score_roi_grid(roi_rgb, model, transform, bank_tensor, device)
    return score, roi_xyxy


def draw_label(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 8)
    x0 = max(0, min(x, img.shape[1] - tw - 10))
    cv2.rectangle(img, (x0, y0), (x0 + tw + 10, y0 + th + baseline + 8), color, -1)
    cv2.putText(img, text, (x0 + 5, y0 + th + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_prediction(
    img: np.ndarray,
    roi_xyxy: Tuple[int, int, int, int],
    score: float,
    threshold: float,
    pad_cls: int,
) -> None:
    x1, y1, x2, y2 = roi_xyxy
    is_unsafe = score >= threshold
    color = (0, 0, 255) if is_unsafe else (0, 180, 0)
    label = "UNSAFE" if is_unsafe else "SAFE"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    cv2.line(img, (xm, y1), (xm, y2), (0, 255, 255), 2)
    cv2.line(img, (x1, ym), (x2, ym), (0, 255, 255), 2)

    draw_label(img, f"{label} cls={pad_cls} score={score:.4f} thr={threshold:.4f}", x1, y1, color)


def draw_objects(img: np.ndarray, objects: List[Tuple[int, Tuple[int, int, int, int]]]) -> None:
    for cls, (x1, y1, x2, y2) in objects:
        color = (255, 120, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        draw_label(img, f"obj cls={cls}", x1, y1, color)


def fit_to_screen(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def iter_images(image_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in exts:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Show grid-hybrid SSL safe/unsafe ROI predictions on test images.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--bank-meta", type=Path, default=DEFAULT_BANK_META)
    parser.add_argument("--bank-dat", type=Path, default=DEFAULT_BANK_DAT)
    parser.add_argument("--threshold-json", type=Path, default=DEFAULT_THRESHOLD_JSON)
    parser.add_argument("--threshold-mode", choices=("f1", "youden"), default="f1")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--draw-objects", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--delay-ms", type=int, default=0, help="0 waits for a key on each image.")
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=900)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = load_threshold(args.threshold_json, args.threshold_mode, args.threshold)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    roi_pad_ratio = float(ckpt.get("roi_pad_ratio", 0.0))
    model = metriks.ProjectionEncoder(ckpt).to(device).eval()
    bank_tensor = load_bank(args.bank_meta, args.bank_dat, device)

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((metriks.IMG_SIZE, metriks.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEVICE] {device}")
    print(f"[THRESHOLD] {threshold:.6f} mode={args.threshold_mode}")
    print(f"[ROI_PAD_RATIO] {roi_pad_ratio:.4f}")
    print("[KEYS] n/space/enter: next | q/esc: quit | s: save current annotated image")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    with torch.no_grad():
        for img_path in iter_images(args.images):
            label_path = args.labels / f"{img_path.stem}.txt"
            img_bgr = metriks.read_image(str(img_path))
            if img_bgr is None:
                continue

            annotated = img_bgr.copy()
            img_h, img_w = annotated.shape[:2]
            pads, objects = collect_pad_boxes(label_path, img_w, img_h)
            image_counts = {"safe": 0, "unsafe": 0, "invalid": 0}

            if args.draw_objects:
                draw_objects(annotated, objects)

            for pad_cls, pad_box in pads:
                if (pad_box[2] - pad_box[0]) < metriks.MIN_PAD_SIZE or (pad_box[3] - pad_box[1]) < metriks.MIN_PAD_SIZE:
                    image_counts["invalid"] += 1
                    continue
                score, roi_xyxy = score_pad(
                    img_bgr,
                    pad_box,
                    roi_pad_ratio,
                    model,
                    transform,
                    bank_tensor,
                    device,
                )
                if score is None or roi_xyxy is None:
                    image_counts["invalid"] += 1
                    continue
                draw_prediction(annotated, roi_xyxy, score, threshold, pad_cls)
                if score >= threshold:
                    image_counts["unsafe"] += 1
                else:
                    image_counts["safe"] += 1

            header = (
                f"{img_path.name} | pads={len(pads)} safe={image_counts['safe']} "
                f"unsafe={image_counts['unsafe']} invalid={image_counts['invalid']}"
            )
            draw_label(annotated, header, 12, 34, (30, 30, 30))

            display = fit_to_screen(annotated, args.max_width, args.max_height)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(args.delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_dir = args.save_dir or (SCRIPT_DIR / "metriks" / "visualized_test_rois")
                save_dir.mkdir(parents=True, exist_ok=True)
                out_path = save_dir / f"{img_path.stem}_ssl_rois.jpg"
                cv2.imencode(".jpg", annotated)[1].tofile(str(out_path))
                print(f"[SAVED] {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
