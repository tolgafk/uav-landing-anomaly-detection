import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
GRID_DIR = os.path.join(ROOT_DIR, "grid_hybrid")
if GRID_DIR not in sys.path:
    sys.path.insert(0, GRID_DIR)

import metriks


DATASET_DIR = os.environ.get("UAV_DATASET_DIR", r"path\to\2026_data_UAP_UAI")
IMAGE_DIR = os.path.join(DATASET_DIR, "images", "test")
LABEL_DIR = os.path.join(DATASET_DIR, "labels", "test")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

SSL_WEIGHTS = os.path.join(OUTPUT_DIR, "ssl_encoder_wholepad.pth")
BANK_META = os.path.join(OUTPUT_DIR, "memory_bank_zero_margin_meta.npz")
BANK_DAT = os.path.join(OUTPUT_DIR, "memory_bank_zero_margin.dat")

OUT_DIR = os.path.join(SCRIPT_DIR, "metriks")
OUT_JSON = os.path.join(OUT_DIR, "ssl_synthetic_metrics_final.json")
OUT_ROC = os.path.join(OUT_DIR, "ssl_synthetic_roc_final.png")
OUT_HIST = os.path.join(OUT_DIR, "ssl_synthetic_hist_final.png")
OUT_CM = os.path.join(OUT_DIR, "ssl_synthetic_confusion_final.png")
OUT_THRESHOLD = os.path.join(OUTPUT_DIR, "anomaly_threshold_wholepad.json")


@torch.no_grad()
def score_roi_wholepad(
    roi_rgb: np.ndarray,
    model: metriks.ProjectionEncoder,
    transform,
    bank_tensor: torch.Tensor,
    device: str,
) -> float:
    x = transform(roi_rgb).unsqueeze(0).to(device)
    z = model(x)
    sim = z @ bank_tensor.T
    return float((1.0 - sim.max(dim=1).values).item())


def save_plots(scores: np.ndarray, labels: np.ndarray, fpr, tpr, auroc: float, threshold: float, cm: np.ndarray) -> None:
    plt.figure()
    plt.hist(scores[labels == 0], bins=40, alpha=0.7, label="SAFE")
    plt.hist(scores[labels == 1], bins=40, alpha=0.7, label="UNSAFE")
    plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2, label=f"Thr: {threshold:.4f}")
    plt.legend()
    plt.xlabel("Anomaly score = 1 - max cosine similarity")
    plt.ylabel("Count")
    plt.title("Whole-Pad SSL Score Distribution")
    plt.tight_layout()
    plt.savefig(OUT_HIST, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Whole-Pad SSL)")
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
    metriks.set_seed(metriks.SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(SSL_WEIGHTS):
        raise FileNotFoundError(f"Missing checkpoint: {SSL_WEIGHTS}. Run ssl_train_wholepad.py first.")
    if not os.path.exists(BANK_META) or not os.path.exists(BANK_DAT):
        raise FileNotFoundError("Missing memory bank. Run build_ssl_mean_wholepad.py after training.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[DEVICE]", device, flush=True)

    ckpt = torch.load(SSL_WEIGHTS, map_location="cpu")
    roi_pad_ratio = float(ckpt.get("roi_pad_ratio", 0.0))
    model = metriks.ProjectionEncoder(ckpt).to(device).eval()

    meta = np.load(BANK_META)
    n_items = int(meta["n_tiles"])
    feat_dim = int(meta["feat_dim"])
    bank_mem = np.memmap(BANK_DAT, mode="r", dtype=np.float32, shape=(n_items, feat_dim))
    bank_tensor = torch.from_numpy(np.array(bank_mem)).to(device)
    bank_tensor = F.normalize(bank_tensor, dim=1)

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((metriks.IMG_SIZE, metriks.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    rois, test_stats = metriks.collect_clean_pad_rois(IMAGE_DIR, LABEL_DIR, roi_pad_ratio)
    if not rois:
        raise RuntimeError("No clean test pads found for synthetic evaluation.")

    scores = []
    labels = []
    with torch.no_grad():
        for roi_rgb in tqdm(rois, desc="Scoring whole-pad clean/synthetic pads", dynamic_ncols=True):
            safe_score = score_roi_wholepad(roi_rgb, model, transform, bank_tensor, device)
            scores.append(safe_score)
            labels.append(0)

            unsafe_roi = metriks.add_synthetic_occlusion(roi_rgb)
            unsafe_score = score_roi_wholepad(unsafe_roi, model, transform, bank_tensor, device)
            scores.append(unsafe_score)
            labels.append(1)

    scores_np = np.array(scores, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int64)

    auroc = roc_auc_score(labels_np, scores_np)
    fpr, tpr, roc_thresholds = roc_curve(labels_np, scores_np)
    youden_threshold = float(roc_thresholds[int(np.argmax(tpr - fpr))])
    best_f1_threshold = metriks.choose_best_f1_threshold(labels_np, scores_np)
    decision_threshold, threshold_selection = metriks.load_decision_threshold(OUT_THRESHOLD, youden_threshold)

    decision_eval = metriks.evaluate_threshold(labels_np, scores_np, decision_threshold)
    youden_eval = metriks.evaluate_threshold(labels_np, scores_np, youden_threshold)
    best_f1_eval = metriks.evaluate_threshold(labels_np, scores_np, best_f1_threshold)
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
        "pipeline": "wholepad_ssl",
        "checkpoint": os.path.basename(SSL_WEIGHTS),
        "memory_bank": os.path.basename(BANK_DAT),
        "feature_space": ckpt.get("feature_space", "encoder"),
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
                "roi_pad_ratio": roi_pad_ratio,
                "score": "wholepad_1_minus_max_cosine",
            },
            f,
            indent=4,
        )

    save_plots(scores_np, labels_np, fpr, tpr, auroc, decision_threshold, cm)

    print("\n--- FINAL WHOLE-PAD SSL RESULTS ---")
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
