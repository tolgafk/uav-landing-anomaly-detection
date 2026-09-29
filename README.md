# Self-Supervised Anomaly Detection for UAV Landing Pads

This repository accompanies the paper **“Self-Supervised Memory-Bank Anomaly Detection for UAV Landing Pads: Whole-Pad versus Grid-Hybrid Representations under Synthetic Occlusions.”** It contains the paper in IEEE and ACM layouts, the experiment code, and the reported metric summaries.

The study learns a nominal representation of landing-pad regions with SimCLR and scores each query by its nearest cosine similarity to a clean memory bank. It compares:

- **Wholepad SSL:** one 2,048-dimensional encoder representation for the complete pad.
- **Grid-Hybrid SSL:** four fixed tiles, 128-dimensional projected representations, and a maximum-tile anomaly score.

## Main results

The controlled test set contains 630 clean landing-pad crops and 630 paired crops with one synthetic obstruction each.

| Method and operating point | AUROC | Precision | Recall | Specificity | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Wholepad, Youden threshold 0.01419 | 0.9236 | 0.8470 | 0.8698 | 0.8429 | 0.8583 | 0.8563 |
| Grid-Hybrid, stored threshold 0.1400 | 0.7555 | 0.8788 | 0.0921 | 0.9873 | 0.1667 | 0.5397 |
| Grid-Hybrid, best-F1 threshold 0.02190 | 0.7555 | 0.6216 | 0.8603 | 0.4762 | 0.7217 | 0.6683 |

The whole-pad representation ranks the tested synthetic anomalies more reliably. The grid result is highly sensitive to threshold selection. Since feature dimensionality and memory-bank construction also differ, the comparison should not be interpreted as an isolated test of spatial granularity.

## Synthetic obstruction protocol

For every eligible clean pad, the evaluator creates one paired anomalous crop. Obstructions cover 22–65% of the pad and randomly vary geometry, position, aspect ratio, rotation, contrast-aware color, texture, contour, and shadow. The available shapes are rotated rectangles, ellipses, bands, and irregular eight-vertex polygons. Synthetic samples are generated at evaluation time and are not stored in this repository.

## Repository contents

```text
paper/
  ieee_paper.tex, ieee_paper.pdf
  acm_paper.tex, acm_paper.pdf
  figures/                 Workflow, ROC, score, and equation figures
code/
  wholepad/                Whole-pad training, bank construction, evaluation
  grid_hybrid/             Grid training, bank construction, evaluation, viewer
results/
  wholepad/                Metric and threshold JSON summaries
  grid_hybrid/             Metric and threshold JSON summaries
```

The workflow figures in both papers are reproduced from the accompanying thesis. The LaTeX files are the editable publication sources.

## Data

The image and annotation files are intentionally excluded. Obtain the **TEKNOFEST Aviation Artificial Intelligence Dataset** from the [dataset folder cited by the thesis](https://drive.google.com/drive/folders/18_VqLBbyTubVSWAXG_CgmuJWGCx0mcBd), then arrange it as:

```text
2026_data_UAP_UAI/
  images/train/
  images/test/
  labels/train/
  labels/test/
```

Set the dataset root before running a pipeline:

```powershell
$env:UAV_DATASET_DIR = "D:\path\to\2026_data_UAP_UAI"
```

## Environment and execution

Create a Python environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run one pipeline from the repository root:

```powershell
python code\wholepad\ssl_train_wholepad.py
python code\wholepad\build_ssl_mean_wholepad.py
python code\wholepad\metriks_wholepad.py
```

```powershell
python code\grid_hybrid\ssl_train_grid.py
python code\grid_hybrid\build_ssl_mean.py
python code\grid_hybrid\metriks.py
```

Training checkpoints and memory-bank arrays are generated under each pipeline’s `outputs/` directory. These large artifacts are excluded from version control.

## Reproducibility notes

- ResNet-101 is initialized with `weights=None` in the archived experiment code.
- Both experiments use 20 epochs, batch size 8, AdamW, learning rate `3e-4`, temperature `0.2`, zero ROI margin, minimum ROI size 32 pixels, and seed 42.
- The active supervised anomaly penalty is zero, and synthetic anomalies are used only for evaluation.
- Published JSON files are included so the paper values can be checked without the image data or model weights.

## Author

**Tolga Ferhan Küçük**  
[kucuktolgaferhan@gmail.com](mailto:kucuktolgaferhan@gmail.com) · [github.com/tolgafk](https://github.com/tolgafk)

## Citation

Please cite the paper source in `paper/` and the metadata in `CITATION.cff`. This repository is an academic project artifact; no license for reuse has been declared.
