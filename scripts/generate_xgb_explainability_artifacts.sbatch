#!/bin/bash
#SBATCH --job-name=ids-xgb-explain
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/capstone15/reports/ids-xgb-explain_%j.log

set -euo pipefail

cd /home/capstone15

RUN_DIR="${RUN_DIR:-/home/capstone15/reports/full_gpu_hpo_models_20260306_195851}"
TEST_CSV="${TEST_CSV:-/home/capstone15/data/merged/metadata_test.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/xgb_explainability}"

python3 scripts/generate_xgb_explainability_artifacts.py \
  --run-dir "$RUN_DIR" \
  --test-csv "$TEST_CSV" \
  --output-dir "$OUTPUT_DIR"

echo "Explainability artifacts generated at: $OUTPUT_DIR"
