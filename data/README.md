# Data directory

The raw wave-function datasets are not committed to this repository.

Expected layout:

```text
data/
├── wave_functions.csv
└── validation/
    ├── shock_1.csv
    ├── shock_2.csv
    ├── gaussian.csv
    └── shock_4.csv
```

The main dataset must contain at least 113 columns. Columns 0–111 are treated as predictors and column 112 as the target. Validation files use the same layout.

The historical project materials do not provide a stable public download URL, source version or checksum for these large raw files. This repository therefore documents the exact schema and portable file layout without claiming public-download reproducibility of the original data. If the original project files are available, record their provenance and checksums before reporting reproduced results.

The cleaned pipeline in `src/supervised_umap.py` fits preprocessing and supervised UMAP on the training partition only and reuses those fitted transforms for held-out and optional validation data.
