@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=.venv39\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo Could not find Python venv interpreter at .venv or .venv39.
  echo Create one first, then rerun this launcher.
  exit /b 1
)

echo Starting CatBoost-E IDS realtime API at http://127.0.0.1:8000
echo Loading the bundled replay cache for the thesis demo.
echo Keep this terminal open while using the React UI.
echo.

"%PYTHON_EXE%" scripts\ids_realtime_api.py ^
  --host 127.0.0.1 ^
  --port 8000 ^
  --base-run-dir reports\full_gpu_hpo_models_20260306_195851 ^
  --matrix-run-dir reports\full_gpu_hpo_models_20260306_195851_protocol_multimodel_robust_matrix_v1_20260314_112105 ^
  --demo-cache-pickle reports\ids_replay_cache\catboost_e_demo_subset_62fd97ba1080fb8e.pkl ^
  --threshold-policy tuned_thresholds ^
  --inference-device auto ^
  --rows-per-second 5 ^
  --local-top-n 8 ^
  --replay-order interleave-protocol-source ^
  --shuffle-seed 42 ^
  --demo-total-rows 9000 ^
  --demo-attack-ratio 0.7 ^
  --demo-seed 42 ^
  --max-recent-alerts 250

endlocal
