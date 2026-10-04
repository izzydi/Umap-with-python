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

The main dataset should contain at least 113 columns. Columns 0–111 are treated as predictors and column 112 as the target. Validation files use the same layout.

The cleaned pipeline in `src/supervised_umap.py` uses these project-relative paths and applies preprocessing learned from the training data to all held-out data.
