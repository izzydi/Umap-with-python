# Supervised UMAP in Python

A Python notebook exploring **Uniform Manifold Approximation and Projection (UMAP)** for dimensionality reduction and visualization of a high-dimensional classification dataset.

## Project overview

The notebook samples a high-dimensional dataset, preprocesses the predictors, fits a supervised two-dimensional UMAP embedding and visualizes the resulting manifold.

## Repository contents

- [`v_01.ipynb`](v_01.ipynb) — Jupyter notebook containing the complete workflow.

## Methods and tools

The notebook uses:

- `pandas` and `NumPy` for data handling,
- `umap-learn` for supervised UMAP,
- `scikit-learn` for preprocessing and classification utilities,
- `matplotlib` and `umap.plot` for visualization.

The preprocessing pipeline applies a `QuantileTransformer` followed by `StandardScaler`, then fits UMAP with a Manhattan distance metric.

## Data requirements

The notebook references a large local CSV dataset through a machine-specific absolute path. That source dataset is not included in this repository, so the path must be changed before the notebook can be reproduced elsewhere.

## Reproducing the analysis

1. Install Python and Jupyter.
2. Install the required packages, including `pandas`, `numpy`, `scikit-learn`, `matplotlib` and `umap-learn[plot]`.
3. Update the CSV path in `v_01.ipynb` to point to the source dataset.
4. Run the notebook from top to bottom.

## Scope

This repository is a focused dimensionality-reduction experiment intended to demonstrate supervised UMAP and visualization of high-dimensional data.
