#!/bin/bash
#SBATCH --job-name=ids-baselines
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/capstone15/reports/ids-baselines_%j.log

set -euo pipefail

cd /home/capstone15

python3 /home/capstone15/scripts/train_baseline_models_stdlib.py \
  --train-cap-per-class 5000 \
  --val-cap-per-class 2000 \
  --test-cap-per-class 7000
