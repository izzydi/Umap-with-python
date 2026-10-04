"""Supervised UMAP workflow for high-dimensional wave-function data.

Preprocessing is fitted strictly on the training set. The same fitted transform and
trained UMAP model are then applied to held-out and optional validation data.
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
EXPECTED_COLUMNS = TARGET_COLUMN + 1
SAMPLE_SIZE = 2000

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def validate_shape(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Validate the source schema and retain 112 predictors plus the target."""
    if df.shape[1] < EXPECTED_COLUMNS:
        raise ValueError(
            f"{source} must contain at least {EXPECTED_COLUMNS} columns; "
            f"found {df.shape[1]}."
        )
    return df.iloc[:, :EXPECTED_COLUMNS].copy()


def load_dataset(path: Path, *, skiprows: int = 1_700_000, nrows: int = 800_000) -> pd.DataFrame:
    """Load the large source data and validate its expected schema."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. See data/README.md for the expected layout.")
    df = pd.read_csv(path, header=None, skiprows=skiprows, nrows=nrows)
    return validate_shape(df, path)


def load_validation(path: Path) -> pd.DataFrame:
    """Load a validation CSV using the same expected feature/target layout."""
    df = pd.read_csv(path, header=None)
    return validate_shape(df, path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return predictors and target from a validated dataframe."""
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
    df = load_dataset(source_path)

    sample_n = min(SAMPLE_SIZE, len(df))
    if sample_n < 4:
        raise ValueError("The source dataset is too small for a stratified train/test split.")

    sampled = df.sample(n=sample_n, replace=False, random_state=RANDOM_STATE)
    x, y = split_xy(sampled)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Learn all data-dependent preprocessing from training data only.
    n_quantiles = min(1000, len(x_train))
    preprocessor = make_pipeline(
        QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=RANDOM_STATE,
        ),
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
            validation_df = load_validation(validation_path)
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
