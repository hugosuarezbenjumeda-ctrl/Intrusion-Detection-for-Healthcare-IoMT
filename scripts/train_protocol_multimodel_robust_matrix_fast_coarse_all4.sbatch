#!/bin/bash
#SBATCH --job-name=ids-proto-mm-fast-c4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/capstone15/reports/ids-proto-mm-fast-c4_%j.log

set -euo pipefail

echo "=== Protocol Multi-Model Robust Matrix (Fast Coarse All4) ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
nvidia-smi || true

CPU_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
echo "CPU_THREADS=$CPU_THREADS"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

PROJECT_DIR="${PROJECT_DIR:-/home/capstone15}"
BASE_RUN_DIR="${BASE_RUN_DIR:-$PROJECT_DIR/reports/full_gpu_hpo_models_20260306_195851}"
TRAIN_CSV="${TRAIN_CSV:-$PROJECT_DIR/data/merged/metadata_train.csv}"
TEST_CSV="${TEST_CSV:-$PROJECT_DIR/data/merged/metadata_test.csv}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/reports}"
STAGE_DIR="${STAGE_DIR:-/home/capstone15/.jobstage/latest}"
VENV="${VENV:-/home/capstone15/.venvs/ids-robust-venv}"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

if [ "${SKIP_PIP_INSTALL:-1}" = "0" ]; then
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --upgrade numpy pandas scikit-learn xgboost
  python -m pip install --upgrade catboost lightgbm torch
fi

cd "$PROJECT_DIR"

BASE_ARGS=(
  --base-run-dir "$BASE_RUN_DIR"
  --train-csv "$TRAIN_CSV"
  --test-csv "$TEST_CSV"
  --out-root "$OUT_ROOT"
  --stage-mode both
  --coarse-seed 42
  --stability-seeds 43,44,45,46,47,48,49,50
  --stability-split-mode shared_val_seed
  --stability-val-split-seed 42
  --stability-sampling-mode fixed
  --stability-sampling-seed 42
  --stage2-topk-global 3
  --protocols wifi,mqtt,bluetooth
  --models xgboost,catboost
  --coarse-models xgboost,catboost
  --attack-source-mode fixed_xgb
  --family-pack bluetooth_recovery_fallback
  --bluetooth-hardneg-max-fraction 0.35
  --bluetooth-benign-min-train-rows 8000
  --xgb-device cuda
  --sampling-policy adaptive_fraction
  --hardneg-epsilons 0.10
  --coarse-hardneg-train-benign-fraction 0.20
  --coarse-hardneg-train-benign-min 4000
  --coarse-hardneg-train-benign-max 30000
  --coarse-hardneg-train-malicious-fraction 0.08
  --coarse-hardneg-train-malicious-min 4000
  --coarse-hardneg-train-malicious-max 20000
  --coarse-hardneg-val-benign-fraction 0.20
  --coarse-hardneg-val-benign-min 3000
  --coarse-hardneg-val-benign-max 10000
  --coarse-val-malicious-fraction 0.08
  --coarse-val-malicious-min 3000
  --coarse-val-malicious-max 8000
  --stability-hardneg-train-benign-fraction 0.35
  --stability-hardneg-train-benign-min 8000
  --stability-hardneg-train-benign-max 50000
  --stability-hardneg-train-malicious-fraction 0.12
  --stability-hardneg-train-malicious-min 8000
  --stability-hardneg-train-malicious-max 30000
  --stability-hardneg-val-benign-fraction 0.35
  --stability-hardneg-val-benign-min 6000
  --stability-hardneg-val-benign-max 15000
  --stability-val-malicious-fraction 0.12
  --stability-val-malicious-min 6000
  --stability-val-malicious-max 12000
  --hardneg-query-budget 16
  --hardneg-query-max-steps 6
  --hardneg-candidates-per-step 12
  --hardneg-query-max-active-rows-per-step 16384
  --hardneg-query-stagnation-patience 2
  --hardneg-query-stagnation-min-delta 5e-5
  --query-score-batch-rows 4096
  --query-refine-topk 1
  --val-malicious-query-budget 16
  --val-malicious-query-max-steps 6
  --val-malicious-candidates-per-step 12
  --val-malicious-query-max-active-rows-per-step 16384
  --val-malicious-query-stagnation-patience 2
  --val-malicious-query-stagnation-min-delta 5e-5
  --val-malicious-score-batch-rows 4096
  --coarse-hardneg-query-budget 16
  --coarse-hardneg-query-max-steps 6
  --coarse-hardneg-candidates-per-step 12
  --coarse-query-score-batch-rows 4096
  --coarse-val-malicious-query-budget 16
  --coarse-val-malicious-query-max-steps 6
  --coarse-val-malicious-candidates-per-step 12
  --coarse-val-malicious-score-batch-rows 4096
  --stability-hardneg-query-budget 128
  --stability-hardneg-query-max-steps 16
  --stability-hardneg-candidates-per-step 24
  --stability-query-score-batch-rows 4096
  --stability-val-malicious-query-budget 128
  --stability-val-malicious-query-max-steps 16
  --stability-val-malicious-candidates-per-step 24
  --stability-val-malicious-score-batch-rows 4096
  --threshold-grid-size 301
  --protocol-max-train-rows 300000
  --protocol-max-val-rows 80000
  --gate-clean-fpr-max 0.005
  --gate-attacked-benign-fpr-max 0.005
  --gate-adv-malicious-recall-min 0.99
  --threshold-gate-attacked-benign-margin 0.8
  --save-models
  --gate-epsilon 0.10
)

if [ -n "${EXTRA_ARGS:-}" ]; then

  EXTRA_ARR=(${EXTRA_ARGS})
  python -u "$STAGE_DIR/scripts/train_protocol_multimodel_robust_matrix.py" "${BASE_ARGS[@]}" "${EXTRA_ARR[@]}"
else
  python -u "$STAGE_DIR/scripts/train_protocol_multimodel_robust_matrix.py" "${BASE_ARGS[@]}"
fi
