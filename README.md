# Supervised UMAP in Python

A dimensionality-reduction project using **supervised UMAP** to learn and visualize low-dimensional structure in high-dimensional wave-function data.

## Primary workflow

[`src/supervised_umap.py`](src/supervised_umap.py) is the audited implementation. It creates the train/test split before learned preprocessing, fits all transformations on training data only, trains supervised UMAP on the training partition and reuses the same fitted objects for held-out and optional validation data.

## Repository structure

```text
.
├── src/
│   └── supervised_umap.py            # audited workflow
├── data/
│   └── README.md                     # expected raw-data layout
├── archive/
│   ├── README.md
│   └── legacy_supervised_umap_exploration.ipynb
├── outputs/                          # created when the script runs
├── requirements.txt
└── README.md
```

## Methods and tools

The cleaned pipeline uses `pandas`/`NumPy` for data handling, `scikit-learn` for stratified splitting and preprocessing, `umap-learn` for supervised manifold learning and `matplotlib` for saved visualizations.

A `QuantileTransformer` and `StandardScaler` are fitted on training data only. Validation files are schema-checked before transformation, and project-relative paths replace the original machine-specific paths.

## Legacy notebook

The original exploratory notebook is preserved under [`archive/`](archive/) for transparency. It contains historical local paths and preprocessing choices that are not suitable for unbiased evaluation, so it is no longer presented as the recommended implementation.

## Data

Raw data are not included. See [`data/README.md`](data/README.md) for expected filenames and structure.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/supervised_umap.py
```

Generated plots are written to `outputs/`.

## Scope

This repository demonstrates reproducible dimensionality reduction and out-of-sample UMAP transformation for high-dimensional classification data. It is maintained as a portfolio project rather than a production inference service.
