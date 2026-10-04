# Supervised UMAP in Python

A dimensionality-reduction project using **supervised UMAP** to learn and visualize low-dimensional structure in high-dimensional wave-function data.

## Primary workflow

[`src/supervised_umap.py`](src/supervised_umap.py) is the audited implementation. It creates the train/test split before learned preprocessing, fits all transformations on training data only, trains supervised UMAP on the training partition and reuses the same fitted objects for held-out and optional validation data.

## Repository structure

```text
.
├── .github/workflows/ci.yml          # Python 3.12 CI
├── src/
│   └── supervised_umap.py            # audited workflow
├── tests/
│   └── test_smoke.py                 # schema/split smoke tests
├── data/
│   └── README.md                     # expected raw-data layout and provenance note
├── archive/
│   ├── README.md
│   └── legacy_supervised_umap_exploration.ipynb
├── outputs/                          # created when the script runs
├── requirements.txt                 # pinned dependencies
└── README.md
```

## Methods and validation

The cleaned pipeline uses `pandas`/`NumPy` for data handling, `scikit-learn` for stratified splitting and preprocessing, `umap-learn` for supervised manifold learning and `matplotlib` for saved visualizations.

A `QuantileTransformer` and `StandardScaler` are fitted on training data only. Supervised UMAP is then fitted on the transformed training partition with training labels. The held-out and optional validation rows use `transform()` only; they do not influence preprocessing or the learned manifold.

## Reproducibility and CI

Direct Python dependencies are pinned in [`requirements.txt`](requirements.txt). GitHub Actions creates a clean Python 3.12 environment, installs the pinned dependencies, compiles the source and runs synthetic smoke tests on every push and pull request. Those tests do not require the unavailable raw wave-function dataset.

## Legacy notebook

The original exploratory notebook is preserved under [`archive/`](archive/) for transparency. It contains historical local paths and preprocessing choices that are not suitable for unbiased evaluation, so it is no longer presented as the recommended implementation.

## Data

Raw data are not included and the historical materials do not provide a stable public source/version/checksum. See [`data/README.md`](data/README.md) for the exact schema and provenance limitation.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/supervised_umap.py
```

Generated plots are written to `outputs/`.

## Scope

This repository demonstrates reproducible dimensionality reduction and out-of-sample UMAP transformation for high-dimensional classification data. It is maintained as a portfolio project rather than a production inference service.
