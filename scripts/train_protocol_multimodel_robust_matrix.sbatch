#!/bin/bash
#SBATCH --job-name=ids-proto-mm-robust
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/capstone15/reports/ids-proto-mm-robust_%j.log

set -euo pipefail

echo "=== Protocol Multi-Model Robust Matrix ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
nvidia-smi || true

CPU_THREADS="${SLURM_CPUS_PER_TASK:-64}"
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
  --stability-seeds 43,44
  --protocols wifi,mqtt,bluetooth
  --models xgboost,catboost
  --coarse-models xgboost,catboost
  --attack-source-mode fixed_xgb
  --xgb-device cuda
  --sampling-policy adaptive_fraction
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
  --hardneg-epsilons 0.05,0.10
  --hardneg-query-budget 120
  --hardneg-query-max-steps 60
  --hardneg-candidates-per-step 3
  --hardneg-feature-subset-size 3
  --hardneg-query-max-active-rows-per-step 16384
  --query-score-batch-rows 2048
  --query-refine-topk 1
  --val-malicious-query-budget 120
  --val-malicious-query-max-steps 60
  --val-malicious-candidates-per-step 3
  --val-malicious-feature-subset-size 3
  --val-malicious-query-max-active-rows-per-step 16384
  --val-malicious-score-batch-rows 2048
  --gate-clean-fpr-max 0.005
  --gate-attacked-benign-fpr-max 0.005
  --gate-adv-malicious-recall-min 0.99
  --threshold-gate-attacked-benign-margin 0.8
  --save-models
  --gate-epsilon 0.10
  --threshold-grid-size 400
  --stage2-topk-global 6
)

if [ -n "${EXTRA_ARGS:-}" ]; then

  EXTRA_ARR=(${EXTRA_ARGS})
  python -u "$STAGE_DIR/scripts/train_protocol_multimodel_robust_matrix.py" "${BASE_ARGS[@]}" "${EXTRA_ARR[@]}"
else
  python -u "$STAGE_DIR/scripts/train_protocol_multimodel_robust_matrix.py" "${BASE_ARGS[@]}"
fi
