# Data Folder

This dataset is too large for standard GitHub Git storage (many files exceed 100 MB, including multi-GB archives).

Use external storage (Drive/S3/Kaggle/Zenodo) or Git LFS if you need to version large binaries.

The omitted merged files:

- `data/merged/metadata_train.csv`
- `data/merged/metadata_test.csv`

are only needed for rebuilding the replay subset or rerunning training/evaluation pipelines. They are not needed for the bundled CatBoost-E UI demo in this repo, because that demo uses the prebuilt replay cache under `reports/ids_replay_cache/`.
