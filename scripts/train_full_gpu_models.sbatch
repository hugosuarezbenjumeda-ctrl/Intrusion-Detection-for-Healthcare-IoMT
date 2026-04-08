#!/bin/bash
#SBATCH --job-name=ids-full-gpu
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=2-12:00:00
#SBATCH --output=/home/capstone15/reports/ids-full-gpu_%j.log

set -euo pipefail

echo "=== Job bootstrap ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

VENV=/scratch/$USER/ids-gpu-venv
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade numpy pandas scikit-learn xgboost catboost
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124

cd /home/capstone15
python /home/capstone15/scripts/train_full_gpu_models.py \
  --train-csv /home/capstone15/data/ciciomt2024/merged/metadata_train.csv \
  --test-csv /home/capstone15/data/ciciomt2024/merged/metadata_test.csv \
  --out-root /home/capstone15/reports \
  --seed 42
