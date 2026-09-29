# Grid-Hybrid SSL

This pipeline divides every zero-margin UAP/UAI crop into a fixed `2 × 2` grid. It trains a ResNet-101 SimCLR model, stores normalized 128-dimensional projected tile features, and uses the largest tile-level nearest-neighbor distance as the pad anomaly score.

Set `UAV_DATASET_DIR` to the dataset root, then run:

```powershell
python ssl_train_grid.py
python build_ssl_mean.py
python metriks.py
```

Use `visualize_ssl_test_rois.py` after the checkpoints and memory bank have been generated. Large generated artifacts are written to `outputs/` and excluded from Git.
