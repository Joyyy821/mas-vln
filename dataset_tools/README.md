# Dataset Tools

Utilities for packaging MAS-VLN randomized warehouse rollouts for Hugging Face.

## Package

```bash
python -m dataset_tools.package_randomized_warehouse \
  --raw-root experiments/randomized_warehouse \
  --out-root /data/randomized_warehouse_hf \
  --version v0.1.0
```

Use `--append` for future releases. Existing rollout tar files are treated as
immutable; pass `--overwrite-rollout-tars` only when intentionally replacing a
previous package artifact.

Parquet metadata requires `pandas` plus `pyarrow` or `fastparquet`.

## Validate

```bash
python -m dataset_tools.validate_packaged_dataset \
  --root /data/randomized_warehouse_hf
```

## Upload

```bash
hf upload-large-folder USER/randomized-warehouse-rgbd \
  /data/randomized_warehouse_hf \
  --repo-type dataset
```

or:

```bash
python -m dataset_tools.upload_to_hf \
  --dataset-root /data/randomized_warehouse_hf \
  --repo-id USER/randomized-warehouse-rgbd
```

