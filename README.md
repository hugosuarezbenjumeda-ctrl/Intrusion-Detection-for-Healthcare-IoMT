# Medical-IoMT Thesis Evidence Repository

This repository is intended for a reader who is checking the work reported in the thesis. Its purpose is to show:

1. what was built
2. what scripts were used
3. what saved artifacts support the claims and tables in the thesis
4. how to run the retained runnable parts of the project

The repository is therefore organized as an evidence-and-inspection package, not as a full raw-data archive.

## What This Repository Is For

If you are reviewing the thesis, this repository should let you do three things:

1. inspect the retained scripts, reports, and thesis-source files
2. trace major claims and appendix tables back to concrete artifacts
3. run the final CatBoost-E operations demo from this repository

It does not contain the full raw dataset or every intermediate model tree, because many of those files are too large or too noisy for a normal GitHub repository.

## Suggested Review Order

If you want to check the project from top to bottom, use this order:

1. Read this file.
2. Open [ARTIFACT_MAP.md](ARTIFACT_MAP.md).
3. Inspect the report folders listed in the evidence sections below.
4. If you want to see the final system running, follow the `How To Run The Final Demo` section.

The thesis-facing narrative markdown files and generator scripts were moved to the companion working repository and are not retained in this trimmed evidence repo.

## Repository Structure

This is where the main parts of the project are located:

- `README.md`
  Main review guide for the repository.
- `ARTIFACT_MAP.md`
  Maps thesis claims and appendix tables to scripts and saved artifacts.
- `UI_RUNTIME_README.md`
  Short operational instructions for the bundled demo UI/API.
- `scripts/`
  Data preparation, training, robustness, explainability, evaluation, and export scripts.
- `reports/`
  Saved outputs from the retained runs that support the thesis.
- `data/`
  Notes about omitted data files and expected local locations.
- `external/`
  Upstream dataset notes, provenance material, and licensing information.
- `web/ids-react-ui/`
  React frontend for the final IDS operations console.
- `start_ids_realtime_api.bat`
  Windows launcher for the API/backend demo.
- `start_ids_react_ui.bat`
  Windows launcher for the frontend demo.

## What Someone Needs To Run This Repository

Not every part of the repository needs the same setup. The requirements depend on what the reader wants to run.

### Case 1. Inspect The Repository Only

If someone only wants to read the code and saved artifacts, no special setup is required.

They can inspect:

- `scripts/`
- `reports/`
- [ARTIFACT_MAP.md](ARTIFACT_MAP.md)

### Case 2. Run The Final Bundled Demo

To run the retained CatBoost-E operations console from this repository, a reader needs:

- Windows + PowerShell
- Node.js with `npm`
- Python 3
- a virtual environment at the repo root named `.venv39` or `.venv`
- the Python packages in `requirements-core.txt`

For this demo, the repository already includes:

- the React UI source in `web/ids-react-ui/`
- the backend API in `scripts/ids_realtime_api.py`
- the three final CatBoost-E model binaries used by the demo
- the replay cache used by the demo
- the threshold and base-report artifacts the API reads

For this demo, the reader does not need:

- `data/merged/metadata_train.csv`
- `data/merged/metadata_test.csv`
- any raw dataset archives

### Case 3. Run Notebook / Docs Tooling

To run the retained notebook workflow or related document-export tooling, a reader needs:

- Python 3
- the packages in `requirements-docs.txt`

In practice, the critical extra dependency is:

- `python-docx`

`requirements-docs.txt` also includes `jupyter`, because the retained EDA workflow is notebook-based.

### Case 4. Run The Notebook, Streamlit UI, Or Heavier Training Scripts

To run more than the bundled demo, the reader will usually need:

- the packages in `requirements-core.txt`
- the packages in `requirements-docs.txt` if they want the notebook or thesis generators
- the omitted merged CSV files under `data/merged/`
- in some cases, a GPU-capable or HPC-style environment

This applies especially to:

- `scripts/generate_advanced_eda_report.ipynb`
- `scripts/ids_xgb_interpretability_ui.py`
- `scripts/train_full_gpu_models.py`
- `scripts/train_hpo_gpu_models.py`
- `scripts/train_hpo_gpu_models_fprfix.py`
- `scripts/train_hpo_gpu_models_leakguard.py`
- `scripts/train_wifi_robust_hardening.py`
- `scripts/train_wifi_robust_rebalance_matrix.py`
- `scripts/train_protocol_multimodel_robust_matrix.py`

## Library And Tooling Summary

The repository uses three dependency manifests.

### Python Core Stack

`requirements-core.txt` is the main Python dependency file. It covers the retained runtime, evaluation, robustness, and training stack.

It includes packages used across the retained scripts, including:

- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `catboost`
- `lightgbm`
- `torch`
- `matplotlib`
- `streamlit`
- `joblib`

### Python Docs / Notebook Stack

`requirements-docs.txt` adds the extra packages needed for:

- `scripts/generate_advanced_eda_report.ipynb`
- optional document-export tooling used outside this trimmed repo

It currently contains:

- `python-docx`
- `jupyter`

### Frontend Stack

The frontend dependencies are defined in:

- `web/ids-react-ui/package.json`
- `web/ids-react-ui/package-lock.json`

These cover the React/Vite UI used by the final bundled demo.

Important:

- the repository does not include `.venv`, `.venv39`, or `node_modules`
- the repository does not include `dist/`
- those are expected to be created locally by the reviewer if they want to run the code

## Which Libraries Matter For Which Parts

This is the shortest practical mapping from task to dependency.

### For The Final Demo API And UI

Needed:

- `numpy`
- `pandas`
- `catboost`
- Node.js / `npm`

Already bundled in the repository:

- final CatBoost-E `.cbm` files
- replay cache
- threshold CSV
- base metrics JSON

### For The XGBoost Interpretability UI

Needed:

- `numpy`
- `pandas`
- `xgboost`
- `streamlit`

### For The EDA Notebook

Needed:

- `numpy`
- `pandas`
- `matplotlib`
- `jupyter`
- the omitted merged CSV files

### For Full Training / HPO / Matrix Scripts

Needed:

- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `catboost`
- `lightgbm`
- `torch`
- in practice, the omitted merged CSV files

Optional but often important:

- GPU support
- remote Linux / HPC execution for heavier runs

### For Thesis Document Generation

Needed:

- `python-docx`

## What Is Included

This repository includes:

- thesis chapter source files and generator scripts
- core scripts used across the project pipeline
- retained reports and result tables used in the thesis
- the final CatBoost-E operations demo
- the three final CatBoost-E model binaries needed by the bundled demo
- a prebuilt replay cache so the demo can run without the omitted multi-GB merged CSV files

## What Is Omitted

This repository intentionally does not include:

- `data/merged/metadata_train.csv`
- `data/merged/metadata_test.csv`
- the full raw dataset archives
- most intermediate trained-model folders
- most candidate and stability subtrees from the final robust-matrix run
- logs, temporary outputs, `node_modules`, and generated frontend build output

These omissions are deliberate. The aim is to keep the repository focused on verifiable evidence and runnable retained components, without adding large or low-signal clutter.

## Context For The Work

The project concerns intrusion detection for medical IoMT traffic. The retained materials in this repository reflect the progression from:

1. merged metadata-aware train/test construction
2. exploratory data analysis
3. baseline and full-data model training
4. hyperparameter tuning and routed modeling
5. explainability and robustness analysis
6. WiFi-specific hardening
7. final protocol-wise robust model selection
8. final CatBoost-E evaluation and thesis table export

The most important thesis-facing material in this trimmed repo is concentrated in:

- `reports/`
- [ARTIFACT_MAP.md](ARTIFACT_MAP.md)

## How To Check The Thesis Evidence

### 1. Claim-to-Artifact Mapping

Open:

- [ARTIFACT_MAP.md](ARTIFACT_MAP.md)

This file is the quickest way to connect thesis claims and appendix tables to the scripts and retained outputs that support them.

### 3. Code Used In The Project

The main project scripts are under:

- `scripts/`

The most important ones for Chapters 3 and 4 are:

- `scripts/merge_ciciomt_with_metadata.py`
- `scripts/train_baseline_models_stdlib.py`
- `scripts/train_full_gpu_models.py`
- `scripts/train_hpo_gpu_models.py`
- `scripts/train_hpo_gpu_models_fprfix.py`
- `scripts/train_hpo_gpu_models_leakguard.py`
- `scripts/generate_xgb_explainability_artifacts.py`
- `scripts/evaluate_xgb_robustness.py`
- `scripts/train_wifi_robust_hardening.py`
- `scripts/train_wifi_robust_rebalance_matrix.py`
- `scripts/train_protocol_multimodel_robust_matrix.py`
- `scripts/evaluate_catboost_protocol_test_metrics.py`
- `scripts/evaluate_catboost_E_surrogate_attacks.py`
- `scripts/tune_catboost_E_thresholds_surrogate_guard.py`
- `scripts/export_catboost_E_tuned_thesis_tables.py`

### 4. Saved Outputs Supporting The Thesis

The retained evidence is under:

- `reports/`

These are the main folders a reviewer is most likely to need:

- `reports/eda_advanced_20260305_231027/`
- `reports/baseline_models_stdlib_20260305_234858/`
- `reports/full_gpu_models_20260306_001638/`
- `reports/full_gpu_hpo_models_20260306_153556/`
- `reports/full_gpu_hpo_models_20260306_195851/`
- `reports/full_gpu_hpo_models_20260306_195851_wifi_rebalance_matrix_v1_20260309_204053/`
- `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105/`

## Workflow Summary

The repository represents the following project flow.

### Phase 1. Data Construction

Script:

- `scripts/merge_ciciomt_with_metadata.py`

Role:

- builds the merged metadata-aware train/test files used by later stages

### Phase 2. Exploratory Data Analysis

Primary materials:

- `scripts/generate_advanced_eda_report.ipynb`
- `reports/EDA_FOR_MODELING.md`
- `reports/eda_advanced_20260305_231027/`

Role:

- establishes the data understanding that motivated later model and robustness decisions

### Phase 3. Baselines And Initial Full-Data Training

Scripts:

- `scripts/train_baseline_models_stdlib.py`
- `scripts/train_full_gpu_models.py`

Primary evidence:

- `reports/baseline_models_stdlib_20260305_234858/`
- `reports/full_gpu_models_20260306_001638/`

### Phase 4. HPO, Routed Models, And Split Repair

Scripts:

- `scripts/train_hpo_gpu_models.py`
- `scripts/train_hpo_gpu_models_fprfix.py`
- `scripts/train_hpo_gpu_models_leakguard.py`

Primary evidence:

- `reports/full_gpu_hpo_models_20260306_153556/`
- `reports/full_gpu_hpo_models_20260306_195851/`

### Phase 5. Explainability And Robustness

Scripts:

- `scripts/generate_xgb_explainability_artifacts.py`
- `scripts/ids_xgb_interpretability_ui.py`
- `scripts/evaluate_xgb_robustness.py`

Primary evidence:

- `reports/XGBOOST_INTERPRETABILITY_UI.md`
- `reports/full_gpu_hpo_models_20260306_195851/xgb_explainability/`
- `reports/full_gpu_hpo_models_20260306_195851/xgb_robustness_realistic_full_20260308_212054/`

### Phase 6. WiFi Hardening And Family Selection

Scripts:

- `scripts/train_wifi_robust_hardening.py`
- `scripts/train_wifi_robust_rebalance_matrix.py`

Primary evidence:

- `reports/full_gpu_hpo_models_20260306_195851_wifi_robust_v1_20260309_103936/`
- `reports/full_gpu_hpo_models_20260306_195851_wifi_robust_v1_20260309_135449/`
- `reports/full_gpu_hpo_models_20260306_195851_wifi_robust_v1_20260309_180250/`
- `reports/full_gpu_hpo_models_20260306_195851_wifi_rebalance_matrix_v1_20260309_204053/`

### Phase 7. Final Protocol Robust Matrix

Scripts:

- `scripts/train_protocol_multimodel_robust_matrix.py`
- `scripts/consolidate_protocol_multimodel_robust_report.py`

Primary evidence:

- `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105/`
- `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_003108/stability_consistency_summary.csv`

### Phase 8. Final CatBoost-E Evaluation And Thesis Tables

Scripts:

- `scripts/evaluate_catboost_protocol_test_metrics.py`
- `scripts/evaluate_catboost_E_surrogate_attacks.py`
- `scripts/tune_catboost_E_thresholds_surrogate_guard.py`
- `scripts/export_catboost_E_tuned_thesis_tables.py`

Primary evidence location:

- `reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105/`

This folder contains, among other outputs:

- final protocol metrics
- surrogate FGSM and PGD results
- threshold-retuning results
- thesis-exported tables

### Phase 9. Thesis Assembly

Status:

- the thesis-assembly generators were moved to the companion working repository and are not included in this trimmed evidence repo

Purpose:

- the retained reports in this repository are the artifact base those thesis-facing documents were generated from

## How To Run The Final Demo

The repository includes a runnable version of the final CatBoost-E operations console.

### Prerequisites

- Windows + PowerShell
- Node.js with `npm`
- Python 3
- a virtual environment at the repo root named `.venv39` or `.venv`

### Environment Setup

From the repository root:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-docs.txt
```

### Start The Backend

```powershell
.\start_ids_realtime_api.bat
```

This starts the API on `http://127.0.0.1:8000`.

Important:

- this launcher uses the bundled replay cache under `reports/ids_replay_cache/`
- it does not require the omitted `data/merged/metadata_train.csv`
- it does not require the omitted `data/merged/metadata_test.csv`
- it can run on CPU if GPU is unavailable

### Start The Frontend

Open a second terminal in the repository root:

```powershell
.\start_ids_react_ui.bat
```

Then open:

- UI: `http://127.0.0.1:5173`
- API health: `http://127.0.0.1:8000/api/health`

If needed, the shorter operational version is in [UI_RUNTIME_README.md](UI_RUNTIME_README.md).

## Thesis Generator Status

The thesis-facing generator scripts are no longer included in this trimmed evidence repository.

`requirements-docs.txt` is still kept for the notebook stack and related document-export tooling used in the companion working repository.

## How To Run The Retained Tests

From the repository root:

```powershell
python -m unittest discover -s scripts -p "test_*.py"
```

These tests cover retained robustness-related logic, not the entire original research environment.

## Data Notes

The merged CSV files used by training and some evaluation stages are intentionally not committed because they are too large for normal GitHub storage.

Those expected local files are:

- `data/merged/metadata_train.csv`
- `data/merged/metadata_test.csv`

If someone wants to rebuild data-dependent stages, they should consult:

- `data/README.md`
- `external/IoT-Healthcare-Security-Dataset/README.md`
- `scripts/merge_ciciomt_with_metadata.py`

Again, those omitted merged files are not needed to run the bundled demo.

## Long-Running Jobs

Some retained scripts were originally intended for remote Linux or HPC execution.

For those, see:

- the paired `.sbatch` files in `scripts/`
- [CAPSTONE15_REMOTE_JOB_SUBMIT_RUNBOOK.md](CAPSTONE15_REMOTE_JOB_SUBMIT_RUNBOOK.md)

This is mainly relevant for:

- full-data training
- HPO runs
- robustness matrices
- heavier export and evaluation stages

## Reproducibility Notes

This repository is strong on evidence and traceability, but only partial on exact environment reconstruction.

What a reviewer can verify directly from this clone:

- the retained scripts and project structure
- the saved report artifacts
- the mapping from thesis claims and appendix tables to concrete files
- the final bundled CatBoost-E demo

What is not fully self-contained here:

- rebuilding the merged data from raw archives without external data access
- rerunning the full thesis pipeline without the omitted large files
- reproducing the exact original experiment environment from this repository alone

Package note:

- exact versions were recovered and pinned for `numpy`, `pandas`, `xgboost`, `catboost`, `lightgbm`, `matplotlib`, and `python-docx`
- `scikit-learn`, `torch`, `streamlit`, and `joblib` remain unpinned because the exact original versions could not be recovered from the current local environment snapshot
- some saved artifacts under `reports/` still preserve original absolute paths from the source workspace or server run directories; those path strings are historical provenance, not current setup instructions for this repository

## Related Files

- [ARTIFACT_MAP.md](ARTIFACT_MAP.md)
- [UI_RUNTIME_README.md](UI_RUNTIME_README.md)
- [CAPSTONE15_REMOTE_JOB_SUBMIT_RUNBOOK.md](CAPSTONE15_REMOTE_JOB_SUBMIT_RUNBOOK.md)
- `data/README.md`
- `external/IoT-Healthcare-Security-Dataset/README.md`

## Final Note

This repository is meant to let someone checking the thesis answer four questions clearly:

1. what was done
2. where the code is
3. where the supporting evidence is
4. what can still be run directly from this repository

For thesis verification, start with [ARTIFACT_MAP.md](ARTIFACT_MAP.md). For the final system demo, run:

```powershell
.\start_ids_realtime_api.bat
.\start_ids_react_ui.bat
```
