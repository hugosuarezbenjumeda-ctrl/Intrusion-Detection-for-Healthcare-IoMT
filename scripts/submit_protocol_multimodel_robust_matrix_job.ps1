param(
    [string]$HostAlias = "rust",
    [ValidateSet("default", "fast")]
    [string]$Profile = "default",
    [string]$RemoteProjectDir = "/home/capstone15",
    [string]$BaseRunDir = "/home/capstone15/reports/full_gpu_hpo_models_20260306_195851",
    [string]$OutRoot = "/home/capstone15/reports",
    [string]$Venv = "/home/capstone15/.venvs/ids-robust-venv",
    [string]$ExtraArgs = "",
    [ValidateSet(0, 1)]
    [int]$SkipPipInstall = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Quote-BashSingle {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + $Value.Replace("'", "'""'""'") + "'"
}

$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
$scriptsDir = Join-Path $repoRoot "scripts"

switch ($Profile) {
    "default" {
        $sbatchName = "train_protocol_multimodel_robust_matrix.sbatch"
        $logPrefix = "ids-proto-mm-robust"
    }
    "fast" {
        $sbatchName = "train_protocol_multimodel_robust_matrix_fast_coarse_all4.sbatch"
        $logPrefix = "ids-proto-mm-fast-c4"
    }
    default {
        throw "Unsupported profile: $Profile"
    }
}

$localFiles = @(
    (Join-Path $scriptsDir "train_protocol_multimodel_robust_matrix.py"),
    (Join-Path $scriptsDir "evaluate_xgb_robustness.py"),
    (Join-Path $scriptsDir "xgb_protocol_ids_utils.py"),
    (Join-Path $scriptsDir $sbatchName)
)

foreach ($f in $localFiles) {
    if (-not (Test-Path -Path $f -PathType Leaf)) {
        throw "Missing required local file: $f"
    }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$remoteStage = "$RemoteProjectDir/.jobstage/iomt_$ts"
$remoteScriptsDir = "$remoteStage/scripts"

Write-Host "Creating remote stage: $remoteScriptsDir"
& ssh $HostAlias "mkdir -p '$remoteScriptsDir'"
if ($LASTEXITCODE -ne 0) {
    throw "Failed creating remote stage directory on host '$HostAlias'."
}

Write-Host "Uploading files to ${HostAlias}:$remoteScriptsDir/"
$scpArgs = @()
$scpArgs += $localFiles
$scpArgs += "${HostAlias}:${remoteScriptsDir}/"
& scp @scpArgs
if ($LASTEXITCODE -ne 0) {
    throw "File upload failed."
}

$qProject = Quote-BashSingle -Value $RemoteProjectDir
$qStage = Quote-BashSingle -Value $remoteStage
$qBaseRun = Quote-BashSingle -Value $BaseRunDir
$qOutRoot = Quote-BashSingle -Value $OutRoot
$qVenv = Quote-BashSingle -Value $Venv
$qSbatch = Quote-BashSingle -Value $sbatchName
$qLogPrefix = Quote-BashSingle -Value $logPrefix
$qExtra = Quote-BashSingle -Value $ExtraArgs

$remoteSubmitScript = @"
set -euo pipefail

PROJECT_DIR=$qProject
STAGE_DIR=$qStage
BASE_RUN_DIR=$qBaseRun
OUT_ROOT=$qOutRoot
VENV=$qVenv
SBATCH_NAME=$qSbatch
LOG_PREFIX=$qLogPrefix
EXTRA_ARGS=$qExtra
SKIP_PIP_INSTALL=$SkipPipInstall

TRAIN_CSV=`$(find "`$PROJECT_DIR/data" -type f -name metadata_train.csv | head -n 1)
TEST_CSV=`$(find "`$PROJECT_DIR/data" -type f -name metadata_test.csv | head -n 1)

if [ -z "`$TRAIN_CSV" ] || [ -z "`$TEST_CSV" ]; then
  echo "ERROR: could not auto-discover metadata_train.csv/metadata_test.csv under `$PROJECT_DIR/data" >&2
  exit 1
fi

if [ ! -f "`$STAGE_DIR/scripts/`$SBATCH_NAME" ]; then
  echo "ERROR: missing sbatch file `$STAGE_DIR/scripts/`$SBATCH_NAME" >&2
  exit 1
fi

JOB_ID=`$(sbatch --parsable \
  --export=ALL,PROJECT_DIR="`$PROJECT_DIR",STAGE_DIR="`$STAGE_DIR",BASE_RUN_DIR="`$BASE_RUN_DIR",TRAIN_CSV="`$TRAIN_CSV",TEST_CSV="`$TEST_CSV",OUT_ROOT="`$OUT_ROOT",VENV="`$VENV",SKIP_PIP_INSTALL="`$SKIP_PIP_INSTALL",EXTRA_ARGS="`$EXTRA_ARGS" \
  "`$STAGE_DIR/scripts/`$SBATCH_NAME")

echo "JOB_ID=`$JOB_ID"
echo "STAGE_DIR=`$STAGE_DIR"
echo "TRAIN_CSV=`$TRAIN_CSV"
echo "TEST_CSV=`$TEST_CSV"
echo "LOG_FILE=`$OUT_ROOT/`${LOG_PREFIX}_`${JOB_ID}.log"
"@

Write-Host "Submitting sbatch job on $HostAlias ..."
$submitOutput = $remoteSubmitScript | & ssh $HostAlias "bash -s"
if ($LASTEXITCODE -ne 0) {
    throw "Remote sbatch submission failed."
}

$jobIdLine = $submitOutput | Select-String -Pattern '^JOB_ID=' | Select-Object -First 1
$logLine = $submitOutput | Select-String -Pattern '^LOG_FILE=' | Select-Object -First 1
$stageLine = $submitOutput | Select-String -Pattern '^STAGE_DIR=' | Select-Object -First 1

if ($null -eq $jobIdLine) {
    $submitOutput | ForEach-Object { Write-Host $_ }
    throw "Submission returned no JOB_ID."
}

$jobId = (($jobIdLine.ToString() -split '=', 2)[1]).Trim()
$logFile = if ($null -ne $logLine) { (($logLine.ToString() -split '=', 2)[1]).Trim() } else { "" }
$stageDirOut = if ($null -ne $stageLine) { (($stageLine.ToString() -split '=', 2)[1]).Trim() } else { $remoteStage }

$metaDir = Join-Path $repoRoot "reports\remote_submit_meta"
New-Item -ItemType Directory -Path $metaDir -Force | Out-Null
$metaPath = Join-Path $metaDir "protocol_multimodel_submit_${jobId}_$ts.txt"
@(
    "job_id=$jobId"
    "host_alias=$HostAlias"
    "profile=$Profile"
    "stage_dir=$stageDirOut"
    "log_file=$logFile"
    "submitted_at_local=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss zzz'))"
) | Set-Content -Path $metaPath -Encoding UTF8

Write-Host ""
Write-Host "Submitted job successfully."
Write-Host "  JOB_ID:    $jobId"
Write-Host "  PROFILE:   $Profile"
Write-Host "  STAGE_DIR: $stageDirOut"
if ($logFile) {
    Write-Host "  LOG_FILE:  $logFile"
}
Write-Host "  META_FILE: $metaPath"
Write-Host ""
Write-Host "Monitor commands:"
Write-Host "  ssh $HostAlias `"squeue -j $jobId`""
if ($logFile) {
    Write-Host "  ssh $HostAlias `"tail -n 120 $logFile`""
}
