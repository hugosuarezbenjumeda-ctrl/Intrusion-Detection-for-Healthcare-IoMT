# IDS Operations Console

This repo includes a runnable CatBoost-E operations demo.

## What Is Bundled

- React UI source under `web/ids-react-ui/`
- realtime API under `scripts/ids_realtime_api.py`
- launchers:
  - `start_ids_realtime_api.bat`
  - `start_ids_react_ui.bat`
- final CatBoost-E model files for WiFi, MQTT, and Bluetooth
- a prebuilt replay cache under `reports/ids_replay_cache/`

The bundled replay cache avoids the omitted multi-GB merged CSV files.

## Prerequisites

- Windows + PowerShell
- Node.js with `npm`
- `.venv39\Scripts\python.exe` or `.venv\Scripts\python.exe`
- Python packages from `requirements-core.txt`

## Start

From the repo root:

```powershell
.\start_ids_realtime_api.bat
```

In a second terminal:

```powershell
.\start_ids_react_ui.bat
```

Then open:

- UI: `http://127.0.0.1:5173`
- API health: `http://127.0.0.1:8000/api/health`

## Notes

- The API launcher uses `--inference-device auto`, so it will use GPU when available and otherwise fall back to CPU.
- `web/ids-react-ui/node_modules/` and `web/ids-react-ui/dist/` are intentionally not committed.
- The replay cache is for demo use, not for retraining or for rebuilding the final thesis tables.
- Some saved manifests and decision-table fields still preserve original absolute paths from the source workspace or server runs. Treat those as historical provenance, not as current setup instructions for this repo.
