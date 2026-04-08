@echo off
setlocal

cd /d "%~dp0\web\ids-react-ui"

set "PATH=C:\Program Files\nodejs;%PATH%"

if not exist "node_modules" (
  echo Installing React UI dependencies...
  npm install
)

echo Starting IDS Operations Console at http://127.0.0.1:5173
echo Make sure the realtime API is running at http://127.0.0.1:8000 (use start_ids_realtime_api.bat).
echo Keep this terminal open while using the UI.
echo.
npm run dev -- --host 127.0.0.1 --port 5173

endlocal
