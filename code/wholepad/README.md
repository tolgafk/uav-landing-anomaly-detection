# Wholepad SSL

This pipeline crops each UAP/UAI landing pad with zero margin, trains a ResNet-101 encoder with SimCLR, stores one normalized 2,048-dimensional feature per nominal pad, and scores a query as `1 - maximum cosine similarity`.

Set `UAV_DATASET_DIR` to the dataset root, then run:

```powershell
python ssl_train_wholepad.py
python build_ssl_mean_wholepad.py
python metriks_wholepad.py
```

The evaluation imports the shared synthetic obstruction generator from `../grid_hybrid/metriks.py`. Generated checkpoints, memory banks, plots, and thresholds are written locally and excluded from Git.
