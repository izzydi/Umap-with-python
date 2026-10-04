"""Supervised UMAP workflow for high-dimensional wave-function data.

The script keeps preprocessing fitted strictly on the training set, then applies
that fitted transform and the trained UMAP model to held-out and validation data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

RANDOM_STATE = 42
TARGET_COLUMN = 112
SAMPLE_SIZE = 2000

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def load_dataset(path: Path, *, skiprows: int = 1_700_000, nrows: int = 800_000) -> pd.DataFrame:
    """Load the source data and retain 112 predictors plus the target column."""
    df = pd.read_csv(path, header=None, skiprows=skiprows, nrows=nrows)
    if df.shape[1] < TARGET_COLUMN + 1:
        raise ValueError(f"Expected at least {TARGET_COLUMN + 1} columns, found {df.shape[1]}.")
    return df.iloc[:, : TARGET_COLUMN + 1].copy()


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return predictors and target from a prepared dataframe."""
    x = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN].to_numpy()
    return x, y


def plot_embedding(embedding: np.ndarray, labels: np.ndarray, title: str, output_path: Path) -> None:
    """Save a two-dimensional embedding plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=11, cmap="Spectral")
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    source_path = DATA_DIR / "wave_functions.csv"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing {source_path}. See data/README.md for the expected data layout."
        )

    df = load_dataset(source_path)
    sample_n = min(SAMPLE_SIZE, len(df))
    sampled = df.sample(n=sample_n, replace=False, random_state=RANDOM_STATE)
    x, y = split_xy(sampled)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Fit preprocessing on training data only to avoid information leakage.
    preprocessor = make_pipeline(
        QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE),
        StandardScaler(),
    )
    x_train_scaled = preprocessor.fit_transform(x_train)
    x_test_scaled = preprocessor.transform(x_test)

    n_neighbors = min(100, max(2, len(x_train_scaled) - 1))
    manifold = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        metric="manhattan",
        random_state=RANDOM_STATE,
    ).fit(x_train_scaled, y_train)

    train_embedding = manifold.transform(x_train_scaled)
    test_embedding = manifold.transform(x_test_scaled)

    plot_embedding(
        train_embedding,
        y_train,
        "Supervised UMAP — training set",
        OUTPUT_DIR / "umap_train.png",
    )
    plot_embedding(
        test_embedding,
        y_test,
        "Supervised UMAP — held-out test set",
        OUTPUT_DIR / "umap_test.png",
    )

    validation_dir = DATA_DIR / "validation"
    if validation_dir.exists():
        for validation_path in sorted(validation_dir.glob("*.csv")):
            validation_df = pd.read_csv(validation_path, header=None)
            validation_df = validation_df.iloc[:, : TARGET_COLUMN + 1]
            if len(validation_df) > 800:
                validation_df = validation_df.sample(
                    n=800,
                    replace=False,
                    random_state=RANDOM_STATE,
                )

            validation_x, validation_y = split_xy(validation_df)
            validation_scaled = preprocessor.transform(validation_x)
            validation_embedding = manifold.transform(validation_scaled)

            plot_embedding(
                validation_embedding,
                validation_y,
                f"Supervised UMAP — {validation_path.stem}",
                OUTPUT_DIR / f"umap_{validation_path.stem}.png",
            )


if __name__ == "__main__":
    main()
