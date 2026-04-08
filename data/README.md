# Data Folder

This dataset is too large for standard GitHub Git storage (many files exceed 100 MB, including multi-GB archives).

Use external storage (Drive/S3/Kaggle/Zenodo) or Git LFS if you need to version large binaries.

Dataset source used for this project:

- Kaggle: `https://www.kaggle.com/datasets/cyberdeeplearning/ciciomt2024`

The merged train/test files used by many retained scripts are produced by the included merge script:

- `scripts/merge_ciciomt_with_metadata.py`

This is a single entrypoint. It creates both of the merged outputs below:

The omitted merged files:

- `data/merged/metadata_train.csv`
- `data/merged/metadata_test.csv`

are only needed for rebuilding the replay subset or rerunning training/evaluation pipelines. They are not needed for the bundled CatBoost-E UI demo in this repo, because that demo uses the prebuilt replay cache under `reports/ids_replay_cache/`.

Typical usage from the repository root:

```powershell
python scripts/merge_ciciomt_with_metadata.py --root <dataset_root> --out-dir data/merged
```

The script defaults to the output names `metadata_train.csv` and `metadata_test.csv`.
