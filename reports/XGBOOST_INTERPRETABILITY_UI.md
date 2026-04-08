# XGBoost Interpretability + UI Runbook

This runbook covers:
- Generating global/local explanation artifacts for protocol-routed XGBoost IDS models.
- Launching the Streamlit UI for interactive scoring and explainability.

## 1) Generate explainability artifacts

From repository root:

```bash
python3 scripts/generate_xgb_explainability_artifacts.py \
  --run-dir reports/full_gpu_hpo_models_20260306_195851 \
  --test-csv data/merged/metadata_test.csv
```

Default output path:

`reports/full_gpu_hpo_models_20260306_195851/xgb_explainability/`

Main outputs:
- `global_feature_importance.csv`
- `local_case_summary.csv`
- `local_case_contributions.csv`
- `local_case_reference_rows.csv`
- `reference_rows.csv`
- `summary.json`
- `manifest.json`

Optional Slurm submission:

```bash
sbatch scripts/generate_xgb_explainability_artifacts.sbatch
```

You can override paths at submit time:

```bash
RUN_DIR=/home/capstone15/reports/full_gpu_hpo_models_20260306_195851 \
TEST_CSV=/home/capstone15/data/merged/metadata_test.csv \
OUTPUT_DIR=/home/capstone15/reports/full_gpu_hpo_models_20260306_195851/xgb_explainability \
sbatch scripts/generate_xgb_explainability_artifacts.sbatch
```

## 2) Launch the UI

```bash
streamlit run scripts/ids_xgb_interpretability_ui.py
```

In the sidebar:
- Set `Run directory` to the chosen `full_gpu_hpo_models_*` directory.
- Set `Explainability directory` to `<run-dir>/xgb_explainability`.

Tabs:
- `Single Flow`: one-row scoring + local explanation.
- `Batch Scoring`: upload CSV, score all rows, download scored output, inspect local explanations.
- `Global View`: protocol-level feature ranking (gain/weight/mean absolute contribution).
- `Case Library`: TP/TN/FP/FN cases with stored local contribution tables.

## 3) Required Python packages

- `pandas`
- `numpy`
- `xgboost`
- `streamlit`

Install in your project venv before launching UI.
