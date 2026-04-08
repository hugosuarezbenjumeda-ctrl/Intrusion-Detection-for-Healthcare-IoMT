#!/bin/bash
#SBATCH --job-name=ids-wifi-hardening
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/capstone15/reports/ids-wifi-hardening_%j.log

set -euo pipefail

echo "=== WiFi Robust Hardening Job ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

PROJECT_DIR="${PROJECT_DIR:-/home/capstone15}"
BASE_RUN_DIR="${BASE_RUN_DIR:-$PROJECT_DIR/reports/full_gpu_hpo_models_20260306_195851}"
TRAIN_CSV="${TRAIN_CSV:-$PROJECT_DIR/data/merged/metadata_train.csv}"
TEST_CSV="${TEST_CSV:-$PROJECT_DIR/data/merged/metadata_test.csv}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/reports}"
SEED="${SEED:-42}"

VENV="${VENV:-/scratch/$USER/ids-robust-venv}"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade numpy pandas xgboost

cd "$PROJECT_DIR"

python -u scripts/train_wifi_robust_hardening.py \
  --base-run-dir "$BASE_RUN_DIR" \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --out-root "$OUT_ROOT" \
  --seed "$SEED" \
  --xgb-device cuda \
  --hardneg-train-benign-sample 30000 \
  --hardneg-val-benign-sample 12000 \
  --hardneg-epsilons 0.03,0.05,0.10 \
  --hardneg-query-budget 220 \
  --hardneg-query-max-steps 90 \
  --hardneg-candidates-per-step 4 \
  --hardneg-feature-subset-size 3 \
  --hardneg-extra-topk-per-epsilon 2500 \
  --hardneg-weight 2.5 \
  --val-malicious-sample 12000 \
  --val-malicious-query-budget 220 \
  --val-malicious-query-max-steps 90 \
  --val-malicious-candidates-per-step 4 \
  --val-malicious-feature-subset-size 3 \
  --threshold-grid-size 600 \
  --wifi-recall-target 0.995 \
  --threshold-adv-malicious-recall-target 0.995 \
  --threshold-fallback-adv-shortfall-weight 4.0
