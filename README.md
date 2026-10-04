# Supervised UMAP in Python

A dimensionality-reduction project using **supervised UMAP** to learn and visualize low-dimensional structure in high-dimensional wave-function data.

## Project overview

The repository includes a clean, reusable Python pipeline in addition to the original exploratory notebook. The production-style script performs the train/test split **before** fitting preprocessing, learns the transformation from training data only, fits supervised UMAP and then applies the fitted preprocessing and manifold consistently to held-out and external-validation data.

## Repository structure

```text
.
├── src/
│   └── supervised_umap.py
├── data/
│   └── README.md
├── outputs/                         # created when the script runs
├── requirements.txt
├── supervised_umap_exploration.ipynb
└── README.md
```

## Methods and tools

- `pandas` and `NumPy` for data handling,
- `scikit-learn` for splitting and preprocessing,
- `umap-learn` for supervised manifold learning,
- `matplotlib` for saved visualizations.

The cleaned pipeline uses a `QuantileTransformer` and `StandardScaler` fitted on the training set only, followed by a supervised two-dimensional UMAP model with Manhattan distance.

## Reproducibility improvements

The original exploratory work is preserved in [`supervised_umap_exploration.ipynb`](supervised_umap_exploration.ipynb), while the reusable `src/supervised_umap.py` removes the notebook's machine-specific paths and pre-split preprocessing by using project-relative paths and training-only preprocessing.

## Data

Raw data are not included. See [`data/README.md`](data/README.md) for the expected filenames and structure.

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

This repository demonstrates reproducible dimensionality reduction and out-of-sample UMAP transformation for high-dimensional classification data. It is maintained as a portfolio project rather than a packaged production service.
