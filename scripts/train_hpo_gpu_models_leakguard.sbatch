#!/bin/bash
#SBATCH --job-name=ids-hpo-leakguard
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=2-12:00:00
#SBATCH --output=/home/capstone15/reports/ids-hpo-leakguard_%j.log

set -euo pipefail

echo "=== Job bootstrap ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

VENV=/scratch/$USER/ids-hpo-gpu-venv
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade numpy pandas scikit-learn xgboost catboost

if ! python -m pip install --upgrade lightgbm; then
  echo "[WARN] lightgbm install failed; script will continue without LightGBM"
fi
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124

cd /home/capstone15
python /home/capstone15/scripts/train_hpo_gpu_models_leakguard.py \
  --train-csv /home/capstone15/data/ciciomt2024/merged/metadata_train.csv \
  --test-csv /home/capstone15/data/ciciomt2024/merged/metadata_test.csv \
  --out-root /home/capstone15/reports \
  --seed 42 \
  --target-fpr 0.01 \
  --val-mod 5 \
  --coarse-trials-boost 12 \
  --refine-trials-boost 6 \
  --coarse-trials-mlp 8 \
  --refine-trials-mlp 4 \
  --ensemble-iters 700 \
  --tune-max-rows-boost 1200000 \
  --tune-max-rows-mlp 700000 \
  --mlp-tune-epochs 18 \
  --mlp-final-epochs 28 \
  --mlp-patience 4 \
  --n-jobs 24
