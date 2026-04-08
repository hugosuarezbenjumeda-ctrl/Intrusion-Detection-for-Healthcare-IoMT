# CAPSTONE15 Remote Job Submit Runbook (Cycle 3: Protocol Multi-Model Robust Matrix)

Purpose: submit the new from-scratch robust matrix job from local Windows PowerShell to server alias `rust`, monitor it, fetch results, and clean temporary staging safely.

This runbook follows your constraints:
- password entry is manual/interactive,
- server project root is `/home/capstone15`,
- train/test CSV must be auto-discovered under `/home/capstone15/data`,
- temporary staging uses `/home/capstone15/.jobstage/iomt_YYYYMMDD_HHMMSS/scripts`,
- venv is `/home/capstone15/.venvs/ids-robust-venv`,
- no placeholder tokens in commands.

## 0) Local PowerShell: set paths + create stage
```powershell
cd C:\Users\Hugo\Desktop\Thesis\Medical-IoMT-thesis-evidence

$localRepo = "C:\Users\Hugo\Desktop\Thesis\Medical-IoMT-thesis-evidence"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$remoteStage = "/home/capstone15/.jobstage/iomt_$ts"

ssh rust "mkdir -p $remoteStage/scripts"
```

## 1) Local PowerShell: upload required scripts
```powershell
scp "$localRepo\scripts\train_protocol_multimodel_robust_matrix.py" `
    "$localRepo\scripts\consolidate_protocol_multimodel_robust_report.py" `
    "$localRepo\scripts\evaluate_xgb_robustness.py" `
    "$localRepo\scripts\xgb_protocol_ids_utils.py" `
    "rust:$remoteStage/scripts/"

ssh rust "ls -l $remoteStage/scripts"
```

## 2) Open server shell (interactive)
```powershell
ssh rust
```

## 3) Inside server shell: set job variables and verify
```bash
set -euo pipefail
cd /home/capstone15

STAGE_DIR=$(ls -dt /home/capstone15/.jobstage/iomt_* | head -n 1)
TRAIN_CSV=$(find /home/capstone15/data -type f -name metadata_train.csv | head -n 1)
TEST_CSV=$(find /home/capstone15/data -type f -name metadata_test.csv | head -n 1)
PROJECT_DIR=/home/capstone15
BASE_RUN_DIR=/home/capstone15/reports/full_gpu_hpo_models_20260306_195851
OUT_ROOT=/home/capstone15/reports
VENV=/home/capstone15/.venvs/ids-robust-venv

echo "STAGE_DIR=$STAGE_DIR"
echo "TRAIN_CSV=$TRAIN_CSV"
echo "TEST_CSV=$TEST_CSV"
echo "BASE_RUN_DIR=$BASE_RUN_DIR"

test -f "$STAGE_DIR/scripts/train_protocol_multimodel_robust_matrix.py"
test -f "$STAGE_DIR/scripts/consolidate_protocol_multimodel_robust_report.py"
test -f "$STAGE_DIR/scripts/evaluate_xgb_robustness.py"
test -f "$STAGE_DIR/scripts/xgb_protocol_ids_utils.py"
test -f "$TRAIN_CSV"
test -f "$TEST_CSV"
test -d "$BASE_RUN_DIR"
```

## 4) Inside server shell: write sbatch launcher
```bash
cat > "$STAGE_DIR/run_protocol_multimodel_robust_matrix.sbatch" << 'EOF'
#!/bin/bash
#SBATCH --job-name=ids-proto-mm-robust
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/capstone15/reports/ids-proto-mm-robust_%j.log

set -euo pipefail

echo "=== Protocol Multi-Model Robust Matrix ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

PROJECT_DIR="${PROJECT_DIR:-/home/capstone15}"
BASE_RUN_DIR="${BASE_RUN_DIR:-/home/capstone15/reports/full_gpu_hpo_models_20260306_195851}"
TRAIN_CSV="${TRAIN_CSV:-/home/capstone15/data/merged/metadata_train.csv}"
TEST_CSV="${TEST_CSV:-/home/capstone15/data/merged/metadata_test.csv}"
OUT_ROOT="${OUT_ROOT:-/home/capstone15/reports}"
STAGE_DIR="${STAGE_DIR:-/home/capstone15/.jobstage/latest}"
VENV="${VENV:-/home/capstone15/.venvs/ids-robust-venv}"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade numpy pandas scikit-learn xgboost
python -m pip install --upgrade catboost lightgbm torch || true

cd "$PROJECT_DIR"

python -u "$STAGE_DIR/scripts/train_protocol_multimodel_robust_matrix.py" \
  --base-run-dir "$BASE_RUN_DIR" \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  --out-root "$OUT_ROOT" \
  --stage-mode both \
  --coarse-seed 42 \
  --stability-seeds 43,44 \
  --protocols wifi,mqtt,bluetooth \
  --models xgboost,catboost,lightgbm,mlp \
  --xgb-device cuda \
  --hardneg-train-benign-sample 12000 \
  --hardneg-train-malicious-sample 12000 \
  --hardneg-val-benign-sample 4000 \
  --val-malicious-sample 4000 \
  --hardneg-epsilons 0.05,0.10 \
  --hardneg-query-budget 120 \
  --hardneg-query-max-steps 60 \
  --hardneg-candidates-per-step 3 \
  --hardneg-feature-subset-size 3 \
  --val-malicious-query-budget 120 \
  --val-malicious-query-max-steps 60 \
  --val-malicious-candidates-per-step 3 \
  --val-malicious-feature-subset-size 3 \
  --gate-clean-fpr-max 0.005 \
  --gate-attacked-benign-fpr-max 0.005 \
  --gate-adv-malicious-recall-min 0.99 \
  --gate-epsilon 0.10 \
  --threshold-grid-size 400 \
  --stage2-topk-global 6
EOF

chmod +x "$STAGE_DIR/run_protocol_multimodel_robust_matrix.sbatch"
```

## 5) Inside server shell: submit job with explicit exports
```bash
JOB_ID=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",STAGE_DIR="$STAGE_DIR",BASE_RUN_DIR="$BASE_RUN_DIR",TRAIN_CSV="$TRAIN_CSV",TEST_CSV="$TEST_CSV",OUT_ROOT="$OUT_ROOT",VENV="$VENV" \
  "$STAGE_DIR/run_protocol_multimodel_robust_matrix.sbatch")

echo "JOB_ID=$JOB_ID"
echo "LOG=/home/capstone15/reports/ids-proto-mm-robust_${JOB_ID}.log"
```

## 6) Inside server shell: monitor job
```bash
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,JobName,Partition,State,Elapsed,ExitCode
tail -n 120 "/home/capstone15/reports/ids-proto-mm-robust_${JOB_ID}.log"
tail -f "/home/capstone15/reports/ids-proto-mm-robust_${JOB_ID}.log"
```

## 7) Fail-fast troubleshooting branch (inside server shell)
If the job fails fast:
```bash
sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed
LOG_FILE="/home/capstone15/reports/ids-proto-mm-robust_${JOB_ID}.log"
echo "LOG_FILE=$LOG_FILE"
tail -n 220 "$LOG_FILE"
grep -nE "ModuleNotFoundError|ImportError|FileNotFoundError|No such file|XGBoostError|CUDA|RuntimeError|Traceback" "$LOG_FILE" || true
```

Common root causes + corrected resubmit:

1) Missing Python package (`ModuleNotFoundError`):
```bash
source "$VENV/bin/activate"
python -m pip install --upgrade numpy pandas scikit-learn xgboost catboost lightgbm torch
JOB_ID=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",STAGE_DIR="$STAGE_DIR",BASE_RUN_DIR="$BASE_RUN_DIR",TRAIN_CSV="$TRAIN_CSV",TEST_CSV="$TEST_CSV",OUT_ROOT="$OUT_ROOT",VENV="$VENV" \
  "$STAGE_DIR/run_protocol_multimodel_robust_matrix.sbatch")
echo "RESUBMITTED_JOB_ID=$JOB_ID"
```

2) Wrong data path (`FileNotFoundError` on train/test CSV):
```bash
TRAIN_CSV=$(find /home/capstone15/data -type f -name metadata_train.csv | head -n 1)
TEST_CSV=$(find /home/capstone15/data -type f -name metadata_test.csv | head -n 1)
echo "TRAIN_CSV=$TRAIN_CSV"
echo "TEST_CSV=$TEST_CSV"
JOB_ID=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",STAGE_DIR="$STAGE_DIR",BASE_RUN_DIR="$BASE_RUN_DIR",TRAIN_CSV="$TRAIN_CSV",TEST_CSV="$TEST_CSV",OUT_ROOT="$OUT_ROOT",VENV="$VENV" \
  "$STAGE_DIR/run_protocol_multimodel_robust_matrix.sbatch")
echo "RESUBMITTED_JOB_ID=$JOB_ID"
```

3) Wrong base run dir (missing baseline artifacts like `best_hparams.json`):
```bash
ls -ld /home/capstone15/reports/full_gpu_hpo_models_*
BASE_RUN_DIR=/home/capstone15/reports/full_gpu_hpo_models_20260306_195851
test -f "$BASE_RUN_DIR/best_hparams.json"
JOB_ID=$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="$PROJECT_DIR",STAGE_DIR="$STAGE_DIR",BASE_RUN_DIR="$BASE_RUN_DIR",TRAIN_CSV="$TRAIN_CSV",TEST_CSV="$TEST_CSV",OUT_ROOT="$OUT_ROOT",VENV="$VENV" \
  "$STAGE_DIR/run_protocol_multimodel_robust_matrix.sbatch")
echo "RESUBMITTED_JOB_ID=$JOB_ID"
```

## 8) After completion (inside server shell): locate run dir + consolidate
```bash
RUN_DIR=$(ls -dt /home/capstone15/reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_* | head -n 1)
echo "RUN_DIR=$RUN_DIR"

source "$VENV/bin/activate"
python "$STAGE_DIR/scripts/consolidate_protocol_multimodel_robust_report.py" --run-dir "$RUN_DIR"
```

## 9) Local PowerShell: copy final results folder + log back
```powershell
cd C:\Users\Hugo\Desktop\Thesis\Medical-IoMT-thesis-evidence

$pullTs = Get-Date -Format "yyyyMMdd_HHmmss"
$dest = "C:\Users\Hugo\Desktop\Thesis\Medical-IoMT-thesis-evidence\reports\server_pull_$pullTs"
New-Item -ItemType Directory -Path $dest -Force | Out-Null

$remoteRun = (ssh rust 'ls -dt /home/capstone15/reports/full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_* | head -n 1').Trim()
$remoteLog = (ssh rust 'ls -t /home/capstone15/reports/ids-proto-mm-robust_*.log | head -n 1').Trim()

scp -r "rust:$remoteRun" "$dest\"
scp "rust:$remoteLog" "$dest\"

Write-Host "Pulled run dir: $remoteRun"
Write-Host "Pulled log: $remoteLog"
Write-Host "Local destination: $dest"
```

## 10) Local PowerShell: cleanup staged temp folder
```powershell
ssh rust "rm -rf $remoteStage"
ssh rust "test -d $remoteStage && echo still_exists || echo removed"
```

## Quoting safety note (PowerShell)
When running `ssh rust '... | head -n 1'` from local PowerShell, use single quotes around the remote command to prevent PowerShell from trying to interpret Linux commands like `head`.
