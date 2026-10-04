# Legacy exploration

`legacy_supervised_umap_exploration.ipynb` is preserved as a historical record of the original exploratory work.

It is **not** the recommended reproducible workflow. The legacy notebook contains machine-specific data paths and exploratory preprocessing choices in which transformations were fitted before the train/test split or re-fitted on validation data.

Use [`../src/supervised_umap.py`](../src/supervised_umap.py) for the audited workflow. That implementation fits preprocessing and supervised UMAP on training data only, then reuses the fitted objects unchanged for held-out and validation data.
