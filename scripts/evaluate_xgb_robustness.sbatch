#!/bin/bash
#SBATCH --job-name=ids-xgb-robust
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/capstone15/reports/ids-xgb-robust_%j.log

set -euo pipefail

cd /home/capstone15

RUN_DIR="${RUN_DIR:-/home/capstone15/reports/full_gpu_hpo_models_20260306_195851}"
TRAIN_CSV="${TRAIN_CSV:-/home/capstone15/data/merged/metadata_train.csv}"
TEST_CSV="${TEST_CSV:-/home/capstone15/data/merged/metadata_test.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/xgb_robustness}"
REALISTIC_MODE="${REALISTIC_MODE:-on}"
XGB_DEVICE="${XGB_DEVICE:-cuda}"
QUERY_BUDGET_MALICIOUS="${QUERY_BUDGET_MALICIOUS:-300}"
QUERY_BUDGET_BENIGN="${QUERY_BUDGET_BENIGN:-150}"
REALISTIC_ATTACK_PER_PROTOCOL="${REALISTIC_ATTACK_PER_PROTOCOL:-5000}"
REALISTIC_BENIGN_PER_PROTOCOL="${REALISTIC_BENIGN_PER_PROTOCOL:-5000}"

python3 scripts/evaluate_xgb_robustness.py \
  --run-dir "$RUN_DIR" \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --realistic-mode "$REALISTIC_MODE" \
  --xgb-device "$XGB_DEVICE" \
  --query-budget-malicious "$QUERY_BUDGET_MALICIOUS" \
  --query-budget-benign "$QUERY_BUDGET_BENIGN" \
  --realistic-sample-attack-per-protocol "$REALISTIC_ATTACK_PER_PROTOCOL" \
  --realistic-sample-benign-per-protocol "$REALISTIC_BENIGN_PER_PROTOCOL"

echo "Robustness artifacts generated at: $OUTPUT_DIR"
